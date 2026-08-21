"""Benchmark integration contract.

A benchmark is onboarded **once, in xrlenv** (task images + oracle cleanliness).
beagle *consumes* that onboarding rather than redoing it. A :class:`Benchmark` is
just three small, orthogonal pluggables:

* **source** (:class:`TaskSource`) — where tasks + per-task environment come from.
  The default (:class:`~beagle.benchmarks.source.HarborCache`) reads xrlenv's
  harbor cache directly, so a benchmark green in xrlenv needs no task code here.
* **harness** (:class:`BenchmarkHarness`) — how the rollout runs, through the
  benchmark's *native* driver (harbor trial / docker drop-in). Never a re-impl.
* **grader** (:class:`Grader`) — how outputs become scores. An **open interface**,
  not a fixed taxonomy: reusable implementations (in-band, patch-eval) are provided
  as building blocks, and any new shape is just another ``Grader``.

Most harbor-family benchmarks are a two-line :class:`~beagle.benchmarks.harness.HarborBenchmark`
subclass (default source + default harness + in-band grader = **zero** custom code).
Only what a benchmark irreducibly owns — its task→prompt mapping and its grading —
is ever written here, and only when the defaults don't fit.
"""

from __future__ import annotations

import json
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from beagle.types import RolloutStatus, Task, TaskContext, TaskResult

if TYPE_CHECKING:
    from beagle.agents.core.base import Runnable
    from beagle.config import RetryPolicy
    from beagle.rollout.binding import RolloutBinding
    from beagle.rollout.runtime import ContainerRuntime


@dataclass
class BenchmarkSpec:
    """Config selecting and filtering a benchmark's tasks.

    ``task_ids=None`` means the full set; a list restricts (and orders) the
    selection; ``exclude_task_ids`` is applied afterwards. ``num_samples > 1`` expands
    each task into N rollouts (pass@k). Task selection is by id — there is deliberately
    no ``limit``/first-N knob. The image fields
    (``namespace`` / ``tag`` / ``registry`` / ``image``) are harbor-family knobs;
    ``dataset`` names the task source location (an HF id, a cache dir, or a checkout).
    """

    name: str
    dataset: str | None = None
    split: str | None = None
    task_ids: list[str] | None = None
    exclude_task_ids: list[str] | None = None
    num_samples: int = 1
    namespace: str | None = None
    tag: str = "main"
    registry: str | None = None
    image: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GradeReport:
    """Outcome of grading a batch. Points at the benchmark's own eval-results dir."""

    num_tasks: int = 0
    num_resolved: int = 0
    score: float = 0.0
    eval_dir: Path | None = None
    per_task: dict[str, Any] = field(default_factory=dict)


class TaskSource(ABC):
    """Where a benchmark's tasks + per-task environment come from."""

    @abstractmethod
    def tasks(self, spec: BenchmarkSpec) -> Iterator[tuple[Task, TaskContext]]:
        """Stream normalized ``(Task, TaskContext)`` for ``spec`` (filtered)."""
        raise NotImplementedError


class _RolloutProgress:
    """A tiny, dependency-free rollout progress line for the per-task fan-out.

    The harbor path shows harbor's own ``Mean:`` bar; a per-task harness (docker drop-in like
    SWE-bench) has no engine, so without this the user stares at a silent terminal for the whole
    patch-generation phase. Prints ``[<benchmark>] rollout N/M · <errs> · <elapsed>`` to stderr —
    in place (``\\r``) on a TTY, else a throttled plain line so a redirected log isn't 500 rows.
    ``update`` is called once per completed task (results arrive in completion order)."""

    def __init__(self, benchmark: str, total: int, parallelism: int = 1, *, stream: Any = None) -> None:
        self._bench = benchmark or "rollout"
        self._total = total
        self._stream = stream if stream is not None else sys.stderr
        self._done = 0
        self._errors = 0
        self._t0 = time.monotonic()
        self._tty = bool(getattr(self._stream, "isatty", lambda: False)())
        # On a non-TTY, emit ~20 lines total (plus every error) instead of one per task.
        self._step = max(1, total // 20)
        # Announce the phase up front: the line updates on task COMPLETION, so a long first task
        # would otherwise leave the terminal silent — the reported "I saw nothing".
        print(f"[{self._bench}] rollout: {total} task(s), parallelism {parallelism}",
              file=self._stream, flush=True)

    def update(self, result: Any) -> None:
        self._done += 1
        errored = getattr(result, "error", None) is not None
        if errored:
            self._errors += 1
        if not self._tty and not errored and self._done != self._total and self._done % self._step:
            return
        elapsed = int(time.monotonic() - self._t0)
        line = f"[{self._bench}] rollout {self._done}/{self._total}"
        if self._errors:
            line += f" · {self._errors} err"
        line += f" · {elapsed // 60}:{elapsed % 60:02d}"
        if self._tty:
            print(f"\r{line}", end="", file=self._stream, flush=True)
        else:
            print(line, file=self._stream, flush=True)

    def close(self) -> None:
        # Terminate the in-place TTY line so the next print starts on a fresh row.
        if self._tty and self._done:
            print("", file=self._stream, flush=True)


class BenchmarkHarness(ABC):
    """Runs an agent on this benchmark's tasks through its native machinery.

    Two entry points, pick one:

    * Implement :meth:`run` for a **per-task** harness (harbor trial, docker
      drop-in). :meth:`rollout` loops it for you.
    * Override :meth:`rollout` for a **stateful / native-runner** harness that owns
      the batch loop — e.g. reusing a container across tasks, or invoking a
      benchmark's own vendored orchestrator (WAI). The reuse/provisioning
      complexity stays inside the harness (or the vendored runner it calls).
    """

    def run(
        self,
        binding: RolloutBinding,
        task: Task,
        task_ctx: TaskContext,
        *,
        runtime: ContainerRuntime,
    ) -> TaskResult:
        """Execute a single rollout; leave native artifacts on disk and set
        :attr:`TaskResult.artifact_dir` / ``trajectory`` to point at them."""
        raise NotImplementedError("implement run() or override rollout()")

    def rollout(
        self,
        agent: Runnable,
        items: list[tuple[Task, TaskContext]],
        *,
        runtime: ContainerRuntime,
        run_dir: Path,
        parallelism: int = 1,
        retry: RetryPolicy | None = None,
        attempt: int = 0,
        resuming: bool = False,
    ) -> Iterable[TaskResult]:
        """Roll ``agent`` out over ``items`` — the **fan-out for a per-task harness**.

        Concurrency ownership splits by harness shape: a **harbor** benchmark lets harbor's Job
        engine drive it (``HarborHarness`` overrides this); a benchmark with **no** engine of its
        own (docker drop-in like SWE-bench) is on **us**, so this default owns the fan-out. Ported
        from coding-bench's ``runner.executor`` — the pattern proven at 128-way for SWE-bench-Verified:

        * ``parallelism <= 1`` → inline serial (shallow stacks; no concurrency quirk on the strict path).
        * ``parallelism > 1``  → ``ThreadPoolExecutor(max_workers=parallelism)`` over per-task
          :meth:`run`. Threads, not asyncio: each ``run`` blocks on a container subprocess that
          releases the GIL. The ``ContainerRuntime`` is **shared** across workers, so it MUST be
          thread-safe (its Protocol contract). Yields in *completion* order when parallel.

        ``retry.infra`` re-runs a task on an infra-transient before it counts as failed; a per-task
        failure is **captured** as a ``FAILED`` :class:`TaskResult` (never raised), so one bad task
        can't sink a 128-way batch — the Runner records it as an error row. ``attempt`` (the
        content-retry round) is unused here (a fresh ``run`` overwrites nothing), as is ``resuming``
        (a per-task ``run`` overwrites its task dir in place — only engine-owned harnesses like harbor,
        which resume a whole job, need it). Override to own the loop (harbor / native-runner do)."""
        from beagle.rollout.retry import run_with_infra_retry

        attempts = 1 + (retry.infra if retry else 0)
        agent_name = getattr(agent, "name", "") or ""
        # Version = the agent's source ref (the exact code under eval / a candidate branch) — same
        # identity the harbor path records; ``""`` for a source-less agent, not "unknown".
        agent_version = ""
        _src = getattr(agent, "source", None)
        if callable(_src):
            try:
                agent_version = str(_src().ref or "")
            except Exception:  # noqa: BLE001 — no resolvable source ref → just omit the version
                agent_version = ""

        def _one(task: Task, ctx: TaskContext) -> TaskResult:
            try:
                result = run_with_infra_retry(
                    lambda: self.run(agent.rollout_binding(ctx), task, ctx, runtime=runtime),
                    attempts=attempts,
                )
            except Exception as e:  # noqa: BLE001 — capture so the batch completes; error is recorded
                result = TaskResult(task_id=task.task_id, status=RolloutStatus.FAILED,
                                    error=f"{type(e).__name__}: {e}")
            # Persist this task's rollout artifacts (patch + native/ATIF trajectory) under the
            # benchmark subtree and point ``artifact_dir`` there — so a per-task harness (docker
            # drop-in) leaves an inspectable, resume-readable tree like harbor's, and the grader
            # writes its report alongside. Harbor overrides ``rollout`` and writes its own tree.
            result.benchmark = result.benchmark or task.benchmark
            write_rollout_artifacts(result, task, run_dir, agent_name=agent_name,
                                    agent_version=agent_version)
            return result

        # Live progress for this per-task fan-out (harbor has its own bar; this covers the docker
        # drop-in path — SWE-bench — which would otherwise show nothing during patch generation).
        progress = _RolloutProgress(items[0][0].benchmark if items else "", len(items), parallelism)
        try:
            if parallelism <= 1:
                for task, ctx in items:
                    result = _one(task, ctx)
                    progress.update(result)
                    yield result
                return
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="beagle-task") as pool:
                futures = [pool.submit(_one, t, c) for t, c in items]
                for fut in as_completed(futures):
                    result = fut.result()
                    progress.update(result)
                    yield result
        finally:
            progress.close()

    def completed(
        self, items: list[tuple[Task, TaskContext]], *, run_dir: Path
    ) -> list[TaskResult]:
        """Which of ``items`` already have a finished result in this harness's native
        tree under ``run_dir`` — read back as :class:`TaskResult`. This is the resume
        seam: the Runner asks the harness (not a house ledger) what's already done, so
        per-task state stays in the benchmark's own artifacts. Default: none (a harness
        that leaves no re-readable per-task record simply re-runs on resume)."""
        return []


class Grader(ABC):
    """Turns rollout results into scores. **An open interface, not a taxonomy.**

    A grader may read a reward the rollout already produced (in-band), run an
    evaluator on the produced patch, call an LLM judge, aggregate several signals —
    anything. The framework imposes no shape; reusable graders
    (:class:`~beagle.benchmarks.grader.InBandGrader`,
    :class:`~beagle.benchmarks.grader.PatchEvalGrader`) are provided as building
    blocks, and a new benchmark shape is just another subclass — no framework change.

    Graders are reusable objects (two benchmarks can share one) and may hold and
    aggregate sub-graders. Per-benchmark grading deps are imported lazily and gated
    behind optional extras, so core install stays slim.
    """

    @abstractmethod
    def grade(
        self,
        results: list[TaskResult],
        *,
        runtime: ContainerRuntime,
        run_dir: Path,
        parallelism: int = 1,
    ) -> GradeReport:
        """Score ``results``. ``runtime`` is available for graders that need to spin
        evaluator containers (via the xrlenv drop-in); in-band graders ignore it.
        ``parallelism`` bounds concurrent evaluation — a patch-eval grader hands it to its
        evaluator (SWE-bench: swebench's native ``max_workers``); in-band graders ignore it."""
        raise NotImplementedError


class Benchmark(ABC):
    """The per-benchmark integration object registered in the benchmark factory.

    Composed of three pluggables. Subclass :class:`~beagle.benchmarks.harness.HarborBenchmark`
    for the common case (all three defaulted); override any single method otherwise.
    """

    name: ClassVar[str]

    @abstractmethod
    def source(self) -> TaskSource:
        """Where this benchmark's tasks come from."""
        raise NotImplementedError

    @abstractmethod
    def harness(self) -> BenchmarkHarness:
        """The native rollout driver for this benchmark."""
        raise NotImplementedError

    @abstractmethod
    def grader(self) -> Grader:
        """The grader for this benchmark."""
        raise NotImplementedError

    def additional_info_pre(self, task: Task, ctx: TaskContext) -> str | None:
        """Optional benchmark **data** injected BEFORE the task text (layer 3). Default: none.

        Override to surface a benchmark's own dataset facts that would otherwise be dropped —
        e.g. setup context ("the service under test is already running on :8080"). Return a plain
        string (**data only** — never workflow/how-to-work prose; that framing is the agent's).
        Most benchmarks leave this ``None`` and their payload is just the task text."""
        return None

    def additional_info_post(self, task: Task, ctx: TaskContext) -> str | None:
        """Optional benchmark **data** injected AFTER the task text (layer 5). Default: none.

        SWE-bench's ``hints_text`` is the one instance in the tree; every other benchmark leaves
        this ``None``. Same contract as :meth:`additional_info_pre` — data only, never framing."""
        return None

    def load_tasks(self, spec: BenchmarkSpec) -> Iterator[tuple[Task, TaskContext]]:
        """Stream normalized tasks (from :meth:`source`), assembling each one's agent-facing
        :attr:`~beagle.types.Task.instruction` **data payload** = ``additional_info_pre`` +
        raw ``problem_statement`` + ``additional_info_post`` (blank sections dropped). The raw
        ``problem_statement`` is left untouched; beagle never frames — the agent supplies its
        own system prompt + generic instruction. See ``notes/task-prompt-injection.md``."""
        for task, ctx in self.source().tasks(spec):
            task.instruction = self._assemble_instruction(task, ctx)
            yield task, ctx

    def _assemble_instruction(self, task: Task, ctx: TaskContext) -> str:
        """Join the layer-3/4/5 sections (pre-hook, task, post-hook), dropping any that are
        blank, into the single payload the agent receives."""
        sections = [self.additional_info_pre(task, ctx), task.problem_statement,
                    self.additional_info_post(task, ctx)]
        return "\n\n".join(s.strip() for s in sections if s and s.strip())


def write_rollout_artifacts(
    result: TaskResult, task: Task, run_dir: Path | None, *, agent_name: str = "",
    agent_version: str = "",
) -> None:
    """Persist a per-task harness's rollout artifacts under ``run_dir/<benchmark>/<task_id>/`` and
    point ``result.artifact_dir`` there (the grader writes its report into the same dir):

    * ``patch.diff`` — the generated patch, when the agent produced one;
    * ``agent/<native>`` — the raw native trajectory (``result.trajectory_text``), captured
      in-rollout because a docker-drop-in container is torn down before the host can sync it;
    * ``agent/trajectory.json`` — ATIF, converted from that native stream (best-effort: needs the
      optional harbor dep + a registered converter; the raw stream is always the fallback).

    No-op without a ``run_dir`` / ``benchmark`` / ``task_id``. Harbor writes its own native tree and
    never calls this."""
    if run_dir is None or not task.benchmark or not result.task_id:
        return
    task_dir = Path(run_dir) / task.benchmark / result.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    result.artifact_dir = task_dir
    if result.patch:
        (task_dir / "patch.diff").write_text(result.patch, encoding="utf-8")
    agent_dir = task_dir / "agent"
    if result.trajectory_text is not None or result.stderr_text is not None:
        agent_dir.mkdir(parents=True, exist_ok=True)
    if result.stderr_text is not None:
        # Full agent stderr for a failed trial — the recorded one-line `error` is only its first
        # meaningful line, so the real cause lives here (masked otherwise by Node's startup warnings).
        (agent_dir / "stderr.log").write_text(result.stderr_text, encoding="utf-8")
    if result.trajectory_text is not None:
        ref = result.trajectory
        raw_name = ref.path.name if (ref and ref.path and ref.path.name) else "trajectory.raw.json"
        (agent_dir / raw_name).write_text(result.trajectory_text, encoding="utf-8")
        try:  # ATIF is best-effort — the raw stream above is the reliable artifact
            from beagle.benchmarks.trajectory import write_trajectory_json

            write_trajectory_json(
                agent_dir, trajectory_format=(ref.format if ref else ""),
                instruction=task.prompt(), agent_name=agent_name, agent_version=agent_version,
                model_name=None)
        except Exception:  # noqa: BLE001 — missing harbor/converter never fails a rollout
            pass


def write_result_json(result: TaskResult) -> None:
    """Write ``<artifact_dir>/result.json`` — the graded TaskResult (status / reward / resolved /
    tokens / error). This is the per-task record a per-task harness's :meth:`BenchmarkHarness.completed`
    reads back on resume (the source of truth is the harness's own tree, not a house ledger). No-op
    without an ``artifact_dir``."""
    if result.artifact_dir is None:
        return
    result.artifact_dir.mkdir(parents=True, exist_ok=True)
    (result.artifact_dir / "result.json").write_text(json.dumps({
        "task_id": result.task_id, "benchmark": result.benchmark, "status": result.status.value,
        "resolved": result.resolved, "applied": result.applied, "reward": result.reward,
        "num_turns": result.num_turns, "duration_sec": result.duration_sec,
        # Harbor-shaped per-phase spans (environment_setup / agent_setup / agent_execution) so a
        # docker-drop-in task's timing is comparable to a harbor/pier trial's native breakdown.
        "timing": result.timing,
        "tokens": dict(result.tokens), "error": result.error,
    }, indent=2), encoding="utf-8")


def read_result_json(path: str | Path) -> TaskResult | None:
    """Inverse of :func:`write_result_json` (for resume). ``None`` on a missing/malformed file —
    the task simply re-runs. The patch/trajectory live in sibling files, not here."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return TaskResult(
        task_id=d.get("task_id", ""), benchmark=d.get("benchmark", ""),
        status=RolloutStatus(d.get("status") or "completed"),
        resolved=bool(d.get("resolved")), applied=bool(d.get("applied")),
        reward=d.get("reward"), num_turns=d.get("num_turns") or 0,
        duration_sec=d.get("duration_sec") or 0.0, timing=d.get("timing") or {},
        tokens=d.get("tokens") or {},
        error=d.get("error"), artifact_dir=Path(path).parent)


__all__ = [
    "BenchmarkSpec",
    "GradeReport",
    "TaskSource",
    "BenchmarkHarness",
    "Grader",
    "Benchmark",
    "write_rollout_artifacts",
    "write_result_json",
    "read_result_json",
]
