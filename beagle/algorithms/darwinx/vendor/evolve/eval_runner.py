"""Wrap `bash scripts/run_harbor.sh` so the orchestrator stays decoupled from
its CLI shape.

`run_full(subset=...)` runs the configured benchmark, optionally filtered by
the resolved task subset (full → no filter; smoke-10 → 10 includes; etc).
`run_subset(task_names)` runs a small ad-hoc set for mini-eval inside the
loop.

Both return an `EvalResult` parsed from the new job dir's `result.json`.
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INFRASTRUCTURE_FAILURE_PATTERNS = (
    "all predefined address pools have been fully subnetted",
    "failed to create network",
    "cannot create network",
    "cannot connect to the docker daemon",
    "docker compose command failed for environment",
)


class EvalInfrastructureError(RuntimeError):
    """Harbor produced a result, but infra failed before trials could run."""

    def __init__(self, message: str, *, job_dir: Path, failures: list[dict[str, str]]):
        super().__init__(message)
        self.job_dir = job_dir
        self.failures = failures


@dataclass
class EvalResult:
    job_dir: Path
    config_path: Path
    subset: str                       # label that was used (full / smoke-10 / custom:..)
    task_names: list[str]             # the resolved task list (empty == full)
    n_trials: int
    n_errors: int
    score: float                      # mean reward across non-error trials
    passing_tasks: list[str] = field(default_factory=list)   # reward 1.0 trials
    failing_tasks: list[str] = field(default_factory=list)   # reward <1.0 trials + exceptions
    rewards_per_task: dict[str, float] = field(default_factory=dict)  # trial id -> reward
    solved_tasks: list[str] = field(default_factory=list)  # all observed trials passed
    unsolved_tasks: list[str] = field(default_factory=list)  # all observed trials failed
    partially_solved_tasks: list[str] = field(default_factory=list)  # mixed pass/fail trials
    task_outcomes: dict[str, str] = field(default_factory=dict)  # task -> solved/unsolved/partial
    task_rewards: dict[str, float] = field(default_factory=dict)  # task -> min trial reward
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def failed_task_names(self) -> list[str]:
        """Task-level work pool: partially solved first, then fully unsolved."""
        return list(dict.fromkeys(self.partially_solved_tasks + self.unsolved_tasks))

    def __post_init__(self) -> None:
        """Derive task-level fields for legacy direct EvalResult construction."""
        if self.task_outcomes:
            return
        rewards = dict(self.rewards_per_task)
        trial_order = list(rewards)
        for trial_name in self.passing_tasks:
            if trial_name not in rewards:
                trial_order.append(trial_name)
            rewards[str(trial_name)] = max(rewards.get(str(trial_name), -1.0), 1.0)
        for trial_name in self.failing_tasks:
            trial_name = str(trial_name)
            if rewards.get(trial_name, 0.0) >= 1.0:
                continue
            if trial_name not in rewards:
                # Failing trial known only by name (e.g. exception-only) →
                # treat as reward 0.0. A partial reward (0 < r < 1) that the
                # caller explicitly recorded in rewards_per_task is preserved
                # rather than clobbered to 0.0.
                trial_order.append(trial_name)
                rewards[trial_name] = 0.0
        if not rewards:
            return
        (
            self.solved_tasks,
            self.unsolved_tasks,
            self.partially_solved_tasks,
            self.task_outcomes,
            self.task_rewards,
        ) = _classify_task_outcomes(rewards, trial_order)
        self.rewards_per_task = rewards


def run_full(
    *,
    config_path: Path,
    cwd: Path,
    subset: str,
    task_names: list[str],
    job_name: str | None = None,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    tee_log_path: Path | None = None,
) -> EvalResult:
    """Run the full harbor benchmark (filtered by `task_names` if non-empty).

    Args:
        config_path: path to a YAML config (e.g. configs/terminal_bench_2.yaml).
        cwd: directory to run from (typically the per-pipeline worktree).
        subset: label for record-keeping (recorded on EvalResult.subset).
        task_names: resolved task list. [] means "no --include filter" → full.
        job_name: optional --job-name override; otherwise harbor uses a
            timestamp.
        extra_env: extra DARWINX_EVAL_* / harbor env vars to merge into the
            subprocess environment.
        tee_log_path: optional file that receives Harbor stdout/stderr as it
            streams, while preserving the same output on this worker's stdout.
    """
    extras: list[str] = []
    for t in task_names:
        extras += ["--include-task-name", t]
    if job_name:
        extras += ["--job-name", job_name]
    extras += list(extra_args or [])

    return _run_harbor(
        config_path=config_path,
        cwd=cwd,
        subset=subset,
        task_names=task_names,
        extras=extras,
        extra_env=extra_env,
        tee_log_path=tee_log_path,
    )


def run_subset(
    *,
    config_path: Path,
    cwd: Path,
    task_names: list[str],
    job_name: str | None = None,
    extra_env: dict[str, str] | None = None,
    tee_log_path: Path | None = None,
) -> EvalResult:
    """Same as run_full but always filters to the given task list (mini-eval)."""
    if not task_names:
        raise ValueError("run_subset requires a non-empty task_names list")
    extras: list[str] = []
    for t in task_names:
        extras += ["--include-task-name", t]
    if job_name:
        extras += ["--job-name", job_name]

    return _run_harbor(
        config_path=config_path,
        cwd=cwd,
        subset=f"adhoc:{len(task_names)}tasks",
        task_names=task_names,
        extras=extras,
        extra_env=extra_env,
        tee_log_path=tee_log_path,
    )


def _run_harbor(
    *,
    config_path: Path,
    cwd: Path,
    subset: str,
    task_names: list[str],
    extras: list[str],
    extra_env: dict[str, str] | None,
    tee_log_path: Path | None,
) -> EvalResult:
    cwd = Path(cwd)
    if not cwd.is_dir():
        raise FileNotFoundError(f"eval cwd does not exist: {cwd}")
    if not config_path.is_file():
        raise FileNotFoundError(f"eval config not found: {config_path}")

    env = {**os.environ, **(extra_env or {})}
    # Self-evolve runs multiple Harbor jobs concurrently. `run_harbor.sh`'s
    # preflight `docker network prune` is safe for standalone runs, but it can
    # race another compose startup after the network is created and before the
    # container attaches, producing transient "network ... not found" failures.
    env.setdefault("HARBOR_SKIP_PRUNE", "1")

    last_infra_error: EvalInfrastructureError | None = None
    for attempt in (1, 2):
        attempt_extras = _extras_for_attempt(extras, attempt)
        cmd = ["bash", "scripts/run_harbor.sh", str(config_path)] + attempt_extras

        # Snapshot existing job dirs so we can find the new one after the run.
        jobs_dir = cwd / "jobs"
        before = set()
        if jobs_dir.is_dir():
            before = {p.name for p in jobs_dir.iterdir() if p.is_dir()}

        rc = _run_harbor_command(
            cmd,
            cwd=cwd,
            env=env,
            tee_log_path=tee_log_path,
            jobs_dir=jobs_dir,
            before=before,
        )

        after = set()
        if jobs_dir.is_dir():
            after = {p.name for p in jobs_dir.iterdir() if p.is_dir()}

        new_dirs = sorted(after - before)
        if not new_dirs:
            raise RuntimeError(
                f"harbor exited rc={rc} but no new jobs/<...>/ dir appeared in {jobs_dir}"
            )
        job_dir = jobs_dir / new_dirs[-1]

        # Parse result.json (shape: stats.evals.<key>.{metrics, reward_stats, exception_stats}).
        result_path = job_dir / "result.json"
        if not result_path.is_file():
            raise RuntimeError(
                f"harbor job dir {job_dir} has no result.json (rc={rc}); "
                f"the run probably crashed before writing aggregates."
            )
        raw = json.loads(result_path.read_text())
        result = _parse_result(
            job_dir=job_dir,
            config_path=config_path,
            subset=subset,
            task_names=task_names,
            raw=raw,
        )
        failures = infrastructure_failures(result)
        if not failures:
            return result

        last_infra_error = EvalInfrastructureError(
            _format_infrastructure_failure_message(job_dir, failures),
            job_dir=job_dir,
            failures=failures,
        )
        if attempt == 1 and prune_stale_docker_resources_if_idle(
            reason=f"retrying Harbor infra failure in {job_dir.name}",
        ):
            continue
        raise last_infra_error

    assert last_infra_error is not None
    raise last_infra_error


def _run_harbor_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    tee_log_path: Path | None,
    jobs_dir: Path,
    before: set[str],
) -> int:
    """Run Harbor, optionally teeing combined stdout/stderr to a log file."""
    if tee_log_path is None:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
        return proc.returncode

    tee_log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )
    assert proc.stdout is not None
    completed_seen_at: float | None = None
    post_result_grace_s = _harbor_post_result_grace_s()
    with tee_log_path.open("ab") as log_f:
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready:
                chunk = proc.stdout.read(4096)
                if chunk:
                    log_f.write(chunk)
                    log_f.flush()
                    _write_stdout_bytes(chunk)
                elif proc.poll() is not None:
                    break

            if proc.poll() is not None:
                _drain_harbor_output(proc, log_f)
                break

            completed = _completed_harbor_result_dir(jobs_dir, before)
            if completed is None:
                completed_seen_at = None
                continue
            if completed_seen_at is None:
                completed_seen_at = time.monotonic()
                continue
            if time.monotonic() - completed_seen_at >= post_result_grace_s:
                _terminate_process_group(proc)
                return proc.returncode if proc.returncode is not None else -signal.SIGTERM
    return proc.wait()


def _drain_harbor_output(proc: subprocess.Popen[bytes], log_f: Any) -> None:
    assert proc.stdout is not None
    while True:
        ready, _, _ = select.select([proc.stdout], [], [], 0)
        if not ready:
            return
        chunk = proc.stdout.read(4096)
        if not chunk:
            return
        log_f.write(chunk)
        log_f.flush()
        _write_stdout_bytes(chunk)


def _harbor_post_result_grace_s() -> float:
    raw = os.environ.get("DARWINX_EVAL_HARBOR_POST_RESULT_GRACE_S", "120") or "120"
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 120.0


def _completed_harbor_result_dir(jobs_dir: Path, before: set[str]) -> Path | None:
    if not jobs_dir.is_dir():
        return None
    for name in sorted({p.name for p in jobs_dir.iterdir() if p.is_dir()} - before):
        result_path = jobs_dir / name / "result.json"
        if not result_path.is_file():
            continue
        try:
            raw = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if raw.get("finished_at"):
            return jobs_dir / name
    return None


def _terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _write_stdout_bytes(chunk: bytes) -> None:
    """Mirror Harbor output to worker stdout even when stdout is text-wrapped."""
    stream = getattr(sys.stdout, "buffer", None)
    try:
        if stream is not None:
            stream.write(chunk)
            stream.flush()
        else:
            sys.stdout.write(chunk.decode(errors="replace"))
            sys.stdout.flush()
    except BrokenPipeError:
        # The supervisor may have gone away; keep the worker-side log tee alive.
        return


def _extras_for_attempt(extras: list[str], attempt: int) -> list[str]:
    if attempt <= 1:
        return list(extras)
    out = list(extras)
    for i, arg in enumerate(out):
        if arg == "--job-name" and i + 1 < len(out):
            out[i + 1] = f"{out[i + 1]}_retry{attempt}"
            return out
    return out


def _parse_result(
    *,
    job_dir: Path,
    config_path: Path,
    subset: str,
    task_names: list[str],
    raw: dict[str, Any],
) -> EvalResult:
    stats = raw.get("stats") or {}
    evals = stats.get("evals") or {}
    if not evals:
        return EvalResult(
            job_dir=job_dir,
            config_path=config_path,
            subset=subset,
            task_names=list(task_names),
            n_trials=int(stats.get("n_trials") or 0),
            n_errors=int(stats.get("n_errors") or 0),
            score=0.0,
            raw=raw,
        )

    # There's typically exactly one eval key (e.g. "monet__claude-opus-4-6__terminal-bench").
    eval_key = next(iter(evals))
    eval_block = evals[eval_key] or {}
    metrics = eval_block.get("metrics") or [{}]
    score = float(metrics[0].get("mean") or 0.0)

    reward_stats = (eval_block.get("reward_stats") or {}).get("reward") or {}
    exception_stats = eval_block.get("exception_stats") or {}

    # Trial names look like "<task>__<6charhash>". We use the trial name as
    # the canonical id (matches the existing rerun script + analyze reports).
    passing_trials: list[str] = list(reward_stats.get("1.0") or [])

    failing_trials: list[str] = []
    for bucket, trials in reward_stats.items():
        if bucket == "1.0":
            continue
        failing_trials.extend(trials or [])
    for _exc_type, trials in exception_stats.items():
        failing_trials.extend(trials or [])
    # De-dupe while preserving order.
    failing_trials = list(dict.fromkeys(failing_trials))

    # Trial-level reward map (taking max if a trial appears in multiple
    # buckets, which can happen after rerun-failed.py merges). Exception-only
    # trials are treated as reward 0.0 unless a post-retry reward 1.0 exists.
    per_trial: dict[str, float] = {}
    trial_order: list[str] = []

    def record_trial(trial_name: str, reward: float) -> None:
        if trial_name not in per_trial:
            trial_order.append(trial_name)
        per_trial[trial_name] = max(per_trial.get(trial_name, -1.0), reward)

    for bucket, trials in reward_stats.items():
        try:
            r = float(bucket)
        except ValueError:
            continue
        for t in trials or []:
            record_trial(str(t), r)
    for _exc_type, trials in exception_stats.items():
        for t in trials or []:
            trial_name = str(t)
            if per_trial.get(trial_name, 0.0) >= 1.0:
                continue
            if trial_name not in per_trial:
                trial_order.append(trial_name)
            per_trial[trial_name] = 0.0

    solved, unsolved, partial, outcomes, task_rewards = _classify_task_outcomes(
        per_trial, trial_order,
    )

    return EvalResult(
        job_dir=job_dir,
        config_path=config_path,
        subset=subset,
        task_names=list(task_names),
        n_trials=int(eval_block.get("n_trials") or stats.get("n_trials") or 0),
        n_errors=int(eval_block.get("n_errors") or stats.get("n_errors") or 0),
        score=score,
        passing_tasks=passing_trials,
        failing_tasks=failing_trials,
        rewards_per_task=per_trial,
        solved_tasks=solved,
        unsolved_tasks=unsolved,
        partially_solved_tasks=partial,
        task_outcomes=outcomes,
        task_rewards=task_rewards,
        raw=raw,
    )


def infrastructure_failures(result: EvalResult) -> list[dict[str, str]]:
    """Return Harbor trial setup failures that should invalidate the eval score."""
    eval_block = _first_eval_block(result.raw)
    exception_stats = eval_block.get("exception_stats") or {}
    failures: list[dict[str, str]] = []
    for exc_type, trials in exception_stats.items():
        for trial in trials or []:
            trial_name = str(trial)
            trial_log = result.job_dir / trial_name / "trial.log"
            text = _read_text_limited(trial_log)
            lowered = text.lower()
            matched = next(
                (p for p in INFRASTRUCTURE_FAILURE_PATTERNS if p in lowered),
                None,
            )
            if matched is None:
                continue
            failures.append({
                "trial": trial_name,
                "exception": str(exc_type),
                "pattern": matched,
                "trial_log": str(trial_log),
            })
    return failures


def _first_eval_block(raw: dict[str, Any]) -> dict[str, Any]:
    evals = (raw.get("stats") or {}).get("evals") or {}
    if not evals:
        return {}
    return next(iter(evals.values())) or {}


def _read_text_limited(path: Path, max_bytes: int = 512 * 1024) -> str:
    try:
        with Path(path).open("rb") as f:
            return f.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _format_infrastructure_failure_message(
    job_dir: Path, failures: list[dict[str, str]],
) -> str:
    examples = ", ".join(f["trial"] for f in failures[:5])
    more = "" if len(failures) <= 5 else f", ... ({len(failures)} total)"
    return (
        f"Harbor eval infrastructure failure in {job_dir}: "
        f"{examples}{more}. The score is invalid and must not be used."
    )


def prune_stale_docker_resources_if_idle(*, reason: str = "") -> bool:
    """Best-effort Docker cleanup, only when no Harbor eval process is active.

    `docker network prune` is safe for detached networks, but it can race a
    compose startup between network creation and container attach. Self-evolve
    therefore only prunes at campaign boundaries or after a Harbor process has
    exited and no sibling Harbor process appears active.
    """
    if os.environ.get("DARWINX_EVAL_SKIP_DOCKER_PRUNE") == "1":
        return False
    if not _docker_available():
        return False
    if _harbor_eval_process_active():
        return False

    if not _run_docker(["docker", "container", "prune", "-f"]):
        return False
    _remove_old_harbor_containers()
    _run_docker(["docker", "network", "prune", "-f"])
    return True


def _docker_available() -> bool:
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _harbor_eval_process_active() -> bool:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,cmd="],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        # Unknown is safer than pruning into an active compose startup.
        return True
    self_pid = os.getpid()
    needles = (
        "scripts/run_harbor.py",
        "scripts/run_harbor.sh",
        "harbor run",
        "terminal-bench",
        "docker compose",
    )
    for line in (getattr(proc, "stdout", "") or "").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == self_pid:
            continue
        cmd = parts[1]
        if any(n in cmd for n in needles):
            return True
    return False


def _remove_old_harbor_containers() -> None:
    deadline = time.monotonic() + _docker_prune_timeout_s()
    age_h = int(os.environ.get("HARBOR_ORPHAN_AGE_H", "1") or "1")
    cutoff_seconds = max(0, age_h * 3600)
    now = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.CreatedAt}}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    for line in (getattr(proc, "stdout", "") or "").splitlines():
        if time.monotonic() >= deadline:
            return
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        cid, name, _created = parts
        if "__" not in name or "-main-" not in name:
            continue
        try:
            inspect = subprocess.run(
                ["docker", "inspect", "--format", "{{.Created}}", cid],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        created_raw = inspect.stdout.strip().replace("Z", "+00:00")
        try:
            created = datetime.fromisoformat(created_raw)
        except ValueError:
            continue
        if (now - created).total_seconds() >= cutoff_seconds:
            remaining = max(1, int(deadline - time.monotonic()))
            _run_docker(["docker", "rm", "-f", cid], timeout_s=min(30, remaining))


def _docker_prune_timeout_s() -> int:
    raw = os.environ.get("HARBOR_DOCKER_PRUNE_TIMEOUT_S", "120") or "120"
    try:
        return max(1, int(raw))
    except ValueError:
        return 120


def _run_docker(cmd: list[str], *, timeout_s: int | None = None) -> bool:
    timeout_s = _docker_prune_timeout_s() if timeout_s is None else max(1, timeout_s)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False

    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return False
    return proc.returncode == 0


def parse_existing_result_dir(job_dir: Path) -> EvalResult:
    """Parse a pre-existing harbor job dir without running anything.

    Used when the user passes `--baseline-logs jobs/<ts>` to skip Phase 2's
    fresh baseline run.
    """
    job_dir = Path(job_dir).resolve()
    result_path = job_dir / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"{job_dir} has no result.json")
    raw = json.loads(result_path.read_text())
    return _parse_result(
        job_dir=job_dir,
        config_path=job_dir / "config.json",
        subset="full",         # caller (orchestrator) overrides if needed
        task_names=[],
        raw=raw,
    )


def restrict_to_subset(
    result: EvalResult,
    *,
    subset_label: str,
    task_names: list[str],
) -> EvalResult:
    """Re-aggregate ``result`` over only the tasks in ``task_names``.

    Used when ``--baseline-logs <full_baseline> --task-subset smoke-10`` is
    given — the full baseline's mean score isn't comparable to the smoke-10
    score, so we recompute over just the subset's tasks.

    Score = mean reward across ``task_names``, treating any subset task that
    didn't appear in the baseline as 0.0 (pessimistic — it needs to be
    solved before we can claim equivalence).

    Returns a new EvalResult; ``result`` is not mutated. ``task_names``
    must be non-empty (caller passes ``cfg.subset_tasks``).
    """
    if not task_names:
        return result

    allowed = set(task_names)
    sub_per_trial = {
        t: r for t, r in result.rewards_per_task.items()
        if _task_base(t) in allowed
    }
    seen = {_task_base(t) for t in sub_per_trial}
    missing = sorted(allowed - seen)

    # Score: mean over the configured subset, with 0.0 fallback for tasks
    # the baseline didn't run.
    rewards: list[float] = list(sub_per_trial.values())
    rewards += [0.0] * len(missing)
    score = sum(rewards) / max(1, len(rewards))

    sub_passing = [t for t in result.passing_tasks if _task_base(t) in allowed]
    sub_failing = [t for t in result.failing_tasks if _task_base(t) in allowed]
    # Synthetic trial entries for missing tasks so they show up as failing.
    sub_failing.extend(missing)
    trial_order = list(sub_per_trial)
    solved, unsolved, partial, outcomes, task_rewards = _classify_task_outcomes(
        sub_per_trial, trial_order, missing_tasks=missing,
    )

    return EvalResult(
        job_dir=result.job_dir,
        config_path=result.config_path,
        subset=subset_label,
        task_names=list(task_names),
        n_trials=len(rewards),
        n_errors=result.n_errors,    # we don't filter exception_stats; coarse but ok
        score=score,
        passing_tasks=sub_passing,
        failing_tasks=sub_failing,
        rewards_per_task=sub_per_trial,
        solved_tasks=solved,
        unsolved_tasks=unsolved,
        partially_solved_tasks=partial,
        task_outcomes=outcomes,
        task_rewards=task_rewards,
        raw=result.raw,
    )


def _classify_task_outcomes(
    rewards_by_trial: dict[str, float],
    trial_order: list[str],
    *,
    missing_tasks: list[str] | None = None,
) -> tuple[list[str], list[str], list[str], dict[str, str], dict[str, float]]:
    by_task: dict[str, list[float]] = {}
    task_order: list[str] = []
    for trial_name in trial_order:
        task = _task_base(trial_name)
        if task not in by_task:
            by_task[task] = []
            task_order.append(task)
        by_task[task].append(float(rewards_by_trial.get(trial_name, 0.0)))

    for task in missing_tasks or []:
        if task in by_task:
            continue
        by_task[task] = [0.0]
        task_order.append(task)

    solved: list[str] = []
    unsolved: list[str] = []
    partial: list[str] = []
    outcomes: dict[str, str] = {}
    task_rewards: dict[str, float] = {}
    for task in task_order:
        rewards = by_task.get(task) or [0.0]
        has_success = any(r >= 1.0 for r in rewards)
        has_failure = any(r < 1.0 for r in rewards)
        task_rewards[task] = min(rewards)
        if has_success and not has_failure:
            solved.append(task)
            outcomes[task] = "solved"
        elif has_failure and not has_success:
            unsolved.append(task)
            outcomes[task] = "unsolved"
        else:
            partial.append(task)
            outcomes[task] = "partially_solved"
    return solved, unsolved, partial, outcomes, task_rewards


def _task_base(trial_name: str) -> str:
    """Strip runner-generated suffixes without corrupting SWE task IDs.

    SWE-bench IDs intentionally contain ``org__issue`` (for example
    ``django__django-14765``), so splitting on the first ``__`` collapses all
    instances in a repository into one pseudo-task. Only strip suffixes the
    runner itself adds: ``__sN`` samples or Harbor's 6–8 character alphanumeric
    trial token.
    """
    import re

    return re.sub(r"__(?:s\d+|[A-Za-z0-9]{6,8})$", "", trial_name)


def task_base(trial_name: str) -> str:
    """Public task-name normalizer used by DB persistence and selection."""
    return _task_base(trial_name)


def task_bases(trial_names: list[str]) -> list[str]:
    """Normalize trial names to task-only names, preserving order."""
    return [task_base(t) for t in trial_names]


__all__ = [
    "EvalResult",
    "run_full",
    "run_subset",
    "parse_existing_result_dir",
    "restrict_to_subset",
    "task_base",
    "task_bases",
]
