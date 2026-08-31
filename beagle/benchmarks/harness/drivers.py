"""Concrete native-harness drivers, one per benchmark-integration shape.

These are the reusable harness implementations that benchmark objects wire up.
Each hands control to the benchmark's *own* driver and only substitutes the
container substrate — the embodiment of the drop-in contract.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from beagle.benchmarks.agent_usage import usage_from_agent_dir
from beagle.benchmarks.base import BenchmarkHarness
from beagle.rollout.binding import GenericBinding, HarborBinding, RolloutBinding
from beagle.rollout.runtime import ContainerRuntime
from beagle.types import RolloutStatus, Task, TaskContext, TaskResult, TrajectoryRef

if TYPE_CHECKING:
    from beagle.agents.core.base import Runnable
    from beagle.config import RetryPolicy


class HarborHarness(BenchmarkHarness):
    """Run tasks through harbor's *own* Job driver.

    Uses ``harbor.Job.create(JobConfig(...))`` + ``job.run()`` — **not** the
    low-level ``SingleStepTrial`` — so harbor writes its native
    ``<jobs_dir>/<job>/<trial>/{agent,verifier,artifacts,config.json,result.json,
    trial.log}`` layout and we read it back byte-compatibly (reward from
    ``trial_result.verifier_result.rewards["reward"]``). Containers are managed by
    harbor's own cluster environment (``xrlenv_plugins.harbor``), so ``runtime`` is
    unused here; ``run_dir`` is harbor's ``jobs_dir``.

    The agent is injected the harbor-native way: its :class:`HarborBinding` supplies
    an ``import_path`` that harbor's ``AgentFactory`` constructs. No per-benchmark
    prompt lives here — harbor's ``instruction.md`` is the task.
    """

    #: The framework package (``harbor``; ``pier`` in :class:`PierHarness`). Pier is a harbor fork
    #: with the same Job/config API, so the driver is parametrized on this one name.
    FRAMEWORK = "harbor"

    #: The xrlenv cluster environment for this framework (reads XRLENV_GRPC_* from env).
    ENV_IMPORT_PATH = "xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster"

    #: The ONE shim the framework imports to run any beagle agent (see M+N note below).
    SHIM_IMPORT_PATH = "beagle.benchmarks.harness._harbor_agent:BeagleInstalledAgent"

    def __init__(self, *, env_import_path: str | None = None) -> None:
        """``env_import_path`` overrides the cluster :attr:`ENV_IMPORT_PATH` for THIS harness — a
        local, non-cluster tb2/harbor run points at its own harbor ``Environment`` here instead of
        monkeypatching the class attribute; falls back to the class default when unset.

        Reach: set it in the run config as ``benchmark.options.env_import_path`` — the runner reads
        a benchmark's ``options`` and passes it to :meth:`Benchmark.harness`, which forwards it here
        (or construct the harness directly). Config, not an env var."""
        if env_import_path:
            self.ENV_IMPORT_PATH = env_import_path

    def _harness_api(self) -> dict[str, Any]:
        """The framework's ``Job`` + config classes, imported lazily from :attr:`FRAMEWORK`.
        harbor and pier expose the same names — only the package (and ``Job``'s location:
        ``harbor.Job`` vs ``pier.job.Job``) differ, so a subclass just sets ``FRAMEWORK``."""
        import importlib

        job_cfg = importlib.import_module(f"{self.FRAMEWORK}.models.job.config")
        trial_cfg = importlib.import_module(f"{self.FRAMEWORK}.models.trial.config")
        root = importlib.import_module(self.FRAMEWORK)
        job = getattr(root, "Job", None) or importlib.import_module(f"{self.FRAMEWORK}.job").Job
        return {
            "Job": job, "JobConfig": job_cfg.JobConfig, "RetryConfig": job_cfg.RetryConfig,
            "AgentConfig": trial_cfg.AgentConfig, "EnvironmentConfig": trial_cfg.EnvironmentConfig,
            "TaskConfig": trial_cfg.TaskConfig,
        }

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
        if not items:
            return []
        AgentConfig = self._harness_api()["AgentConfig"]  # lazy: beagle[terminal-bench] / [deep-swe]

        binding = agent.rollout_binding(items[0][1])
        if isinstance(binding, HarborBinding):
            # Escape hatch: a genuinely pre-existing harbor-native agent brings its
            # own import_path. New agents should NOT need this.
            agent_config = AgentConfig(
                import_path=binding.import_path,
                model_name=binding.model_name or None,
                kwargs=binding.kwargs,
            )
        else:
            # Default M+N path: wrap ANY beagle agent in the one generic shim.
            # Harbor constructs it by import_path and calls agent.run() through a
            # runtime backed by the trial environment — no per-agent harbor class.
            identity = _agent_identity(agent)
            agent_config = AgentConfig(
                import_path=self.SHIM_IMPORT_PATH,
                model_name=identity.get("model"),
                kwargs={"identity": identity},
            )
        # A content-retry round (attempt>0, the Runner re-running unresolved tasks) writes to a
        # sibling `<benchmark>-retry<N>` job dir — a distinct name so harbor's own resume doesn't
        # skip the tasks as already-done. attempt 0 keeps the bare benchmark job name.
        base_name = (items[0][0].benchmark if items else "") or None
        job_name = f"{base_name}-retry{attempt}" if (attempt and base_name) else None
        return self._run_job(items, agent_config, run_dir=run_dir, parallelism=parallelism,
                             retry=retry, job_name=job_name, resuming=resuming)

    def completed(
        self, items: list[tuple[Task, TaskContext]], *, run_dir: Path
    ) -> list[TaskResult]:
        """Read back finished trials from harbor's native tree — ``run_dir/<benchmark>/
        <trial>/result.json`` — as TaskResults, so resume derives done-state from harbor
        (the source of truth) instead of a house ledger.

        Match each expected task PREFERENTIALLY by the full id its ``result.json`` RECORDS, not by
        the trial DIR name: harbor/pier slugify the dir to ``<task[:32]>__<hash>`` (truncated to 32
        chars), so a glob on the full ``task_id`` silently misses every id longer than 32 chars —
        those would be re-run as 'missing' even though they finished. One scan indexes each trial by
        its recorded id (:func:`_harbor_trial_task_id`). For trees that DON'T record identity (older
        runs, minimal fixtures) fall back to the dir name — an exact ``<task_id>/`` then a
        ``<task_id>__*`` glob (harbor's hash suffix on a short, non-truncated id)."""
        import json

        if not items:
            return []
        bench_dir = Path(run_dir) / (items[0][0].benchmark or "")
        index: dict[str, tuple[Path, dict[str, Any]]] = {}
        for rj in sorted(bench_dir.glob("*/result.json")):
            try:
                d = json.loads(rj.read_text())
            except Exception:  # noqa: BLE001 — a malformed trial just re-runs, never breaks resume
                continue
            tid = _harbor_trial_task_id(d)
            if tid and tid not in index:
                index[tid] = (rj, d)
        out: list[TaskResult] = []
        for task, _ctx in items:
            hit = index.get(task.task_id)
            if hit is None:  # fallback: dir-name match for trials with no recorded identity
                exact = bench_dir / task.task_id / "result.json"
                cands = ([exact] if exact.exists()
                         else sorted(bench_dir.glob(f"{task.task_id}__*/result.json")))
                rj = next((c for c in cands if c.exists()), None)
                if rj is None:
                    continue
                try:
                    hit = (rj, json.loads(rj.read_text()))
                except Exception:  # noqa: BLE001 — malformed → re-run, never break resume
                    continue
            rj, d = hit
            try:
                out.append(_result_from_harbor_json(d, task.task_id, rj.parent))
            except Exception:  # noqa: BLE001 — a malformed trial just re-runs, never breaks resume
                continue
        return out

    # -- internals -----------------------------------------------------------

    def _run_job(
        self,
        items: list[tuple[Task, TaskContext]],
        agent_config: Any,
        *,
        run_dir: Path,
        parallelism: int,
        job_name: str | None = None,
        retry: RetryPolicy | None = None,
        resuming: bool = False,
    ) -> list[TaskResult]:
        import asyncio

        api = self._harness_api()  # lazy: beagle[terminal-bench] / [deep-swe]
        JobConfig, RetryConfig = api["JobConfig"], api["RetryConfig"]
        EnvironmentConfig, TaskConfig = api["EnvironmentConfig"], api["TaskConfig"]

        from beagle.rollout.retry import INFRA_RETRY_EXCEPTIONS

        # Name harbor's job dir after the benchmark (a rollout group is one benchmark) —
        # NOT a random uuid. This is a HARBOR-harness detail, not a cross-harness contract:
        # it yields harbor's native `<run_dir>/<benchmark>/<trial>/…`, is self-describing,
        # deterministic (stable path across resume), and lets harbor's OWN resume find the
        # prior job dir (a random name defeated all three). Fall back to a uuid only if a
        # task carries no benchmark. Non-harbor harnesses write their own native layout.
        job_name = job_name or (items[0][0].benchmark if items else "") or f"beagle-{uuid.uuid4().hex[:8]}"
        job_dir = Path(run_dir) / job_name
        want = {t.task_id for t, _ in items}

        # RESUME reconcile (only when the Runner is actually resuming — NOT a plain re-invocation).
        # A prior harbor job already lives here (its ``config.json`` is present). Harbor's native
        # resume is all-or-nothing on the WHOLE job: it (1) refuses a JobConfig that differs from the
        # stored one — so beagle can't hand it a task SUBSET — and (2) skips any trial that still has a
        # ``result.json`` (matched by config; only a *cancelled* trial re-runs). So to re-run exactly
        # the set beagle's ``plan_resume`` chose, REUSE the stored full config (harbor's check passes)
        # and DELETE those tasks' trial dirs (absent → harbor's "remaining" → re-run in place; every
        # other trial is matched → skipped). The re-run subset's fresh results are then read back off
        # the tree (full ids), not the mixed existing+new ``job_result``.
        reconcile = resuming and (job_dir / "config.json").exists()
        if reconcile:
            config = JobConfig.model_validate_json((job_dir / "config.json").read_text())
            _delete_trial_dirs(job_dir, want)
        else:
            task_configs = []
            for task, _ctx in items:
                task_dir = task.extras.get("harbor_task_dir")
                if not task_dir:
                    raise ValueError(
                        f"task {task.task_id!r} has no 'harbor_task_dir' extra — use the "
                        f"HarborCache source so the harbor task path is available."
                    )
                task_configs.append(TaskConfig(path=Path(task_dir), trial_name=task.task_id))

            job_kwargs: dict[str, Any] = dict(
                job_name=job_name,
                jobs_dir=run_dir,
                n_concurrent_trials=parallelism,
                environment=EnvironmentConfig(import_path=self.ENV_IMPORT_PATH),
                agents=[agent_config],
                tasks=task_configs,
            )
            if retry is not None:
                # Infra-only trial retry: harbor's trial queue re-runs a trial in a FRESH container
                # ONLY on an INFRA_RETRY_EXCEPTIONS type (include_exceptions is a whitelist), so a
                # content outcome (agent timeout, verifier failure, rate limit) is never re-rolled.
                if retry.infra > 0:
                    job_kwargs["retry"] = RetryConfig(
                        max_retries=retry.infra,
                        include_exceptions=set(INFRA_RETRY_EXCEPTIONS),
                    )
                if retry.timeout_multiplier != 1.0:
                    job_kwargs["timeout_multiplier"] = retry.timeout_multiplier
            config = JobConfig(**job_kwargs)

        async def _go() -> Any:
            job = await api["Job"].create(config)
            return await job.run()

        job_result = asyncio.run(_go())

        # Honor harbor's filesystem contract: emit agent/trajectory.json (ATIF) per trial. Done HERE,
        # post-job — NOT in the shim — because on the xrlenv cluster the agent's native stream is only
        # synced from the container to the host trial dir AFTER the agent step. On a resume, emit ONLY
        # the re-run trials (the ``job_result`` also carries the skipped existing ones — re-emitting
        # those would blank their instruction).
        trials = list(getattr(job_result, "trial_results", []))
        emit_trials = [tr for tr in trials if _live_trial_task_id(tr) in want] if reconcile else trials
        _emit_trajectories(SimpleNamespace(trial_results=emit_trials), items,
                           agent_config, run_dir=run_dir, job_name=job_name)

        if reconcile:
            # Re-read just the re-run tasks' fresh results from the native tree (full ids).
            return self.completed(items, run_dir=run_dir)
        return [_trial_to_result(tr, jobs_dir=run_dir, job_name=job_name) for tr in trials]


def _error_string(exc: Any) -> str | None:
    """A full, substring-matchable error from harbor's ``ExceptionInfo`` (live object OR
    on-disk dict): ``"<type>: <message>"``. Downstream infra/timeout tolerance greps this
    message (e.g. "fetch failed", "AgentTimeoutError"), so the type name alone isn't enough."""
    if exc is None:
        return None
    if isinstance(exc, dict):
        etype, emsg = exc.get("exception_type") or "", exc.get("exception_message") or ""
    else:
        etype = getattr(exc, "exception_type", "") or ""
        emsg = getattr(exc, "exception_message", "") or ""
    etype, emsg = str(etype).strip(), str(emsg).strip()
    if etype and emsg:
        return f"{etype}: {emsg}"
    return etype or emsg or None


def _agent_error(agent_result: Any) -> str | None:
    """The agent-PROCESS failure recorded at ``agent_result.metadata.error`` — e.g. a monet
    ``exited rc=1: Error: fetch failed`` crash. This is DISJOINT from ``exception_info`` (the
    harbor/pier trial-runner exception): an agent that crashes but whose verifier still scored the
    partial container state lands here with ``exception_info`` UNSET. Folding it into the task error
    keeps an infra crash counted as ERRORED (``num_errored``), not a legitimate ``reward=0``
    capability failure — otherwise agent-crash trials silently depress the pass-rate. Handles both
    the live object (``TrialResult.agent_result``) and the on-disk dict (``result.json``) forms."""
    if agent_result is None:
        return None
    meta = (agent_result.get("metadata") if isinstance(agent_result, dict)
            else getattr(agent_result, "metadata", None))
    err = (meta.get("error") if isinstance(meta, dict) else getattr(meta, "error", None)) if meta else None
    if not err:
        return None
    return str(err).strip() or None


def _tokens_from_harbor(n_in: int | None, n_cache: int | None, n_out: int | None) -> dict[str, int]:
    """Reconstruct beagle's cache-split token dict from harbor's per-trial counters.

    Harbor's ``n_input_tokens`` is total input **including** cache and ``n_cache_tokens`` is the cached
    subset (harbor keeps a single cache counter, not beagle's read/write split), so: ``prompt = n_input``,
    ``cache_read = n_cache`` (the whole cached subset — attributed to reads, the common case; write-cache
    isn't distinguishable through harbor), ``input_uncached = n_input - n_cache``. This preserves the
    invariant ``prompt = input_uncached + cache_read + cache_write`` and keeps ``run.json``'s cache split
    non-zero (the shim must set ``n_cache_tokens`` for it to be populated — older runs left it null, so
    ``cache_read`` reads back 0 there)."""
    prompt = int(n_in or 0)
    cache = min(int(n_cache or 0), prompt)  # cache is a subset of input — never exceed it
    return {
        "prompt": prompt,
        "completion": int(n_out or 0),
        "input_uncached": prompt - cache,
        "cache_read": cache,
        "cache_write": 0,
    }


def _harbor_trial_task_id(d: dict[str, Any]) -> str | None:
    """Recover the FULL task id a harbor/pier trial records in its ``result.json``.

    The trial DIR name can't be trusted for this — harbor/pier slugify it to ``<task[:32]>__<hash>``
    (truncated to 32 chars), so any longer id is unrecoverable from the dir alone. The recorded fields
    keep the full identity: prefer the task **path basename** (the exact source dir the run used),
    then the nested ``config.task.path`` basename, then ``task_name`` (which may be ``"<owner>/<task>"``)."""
    tid = d.get("task_id")
    if isinstance(tid, dict) and tid.get("path"):
        return Path(str(tid["path"])).name or None
    if isinstance(tid, str) and tid:
        return tid
    cfg = d.get("config")
    if isinstance(cfg, dict):
        tpath = (cfg.get("task") or {}).get("path")
        if tpath:
            return Path(str(tpath)).name or None
    name = d.get("task_name")
    if isinstance(name, str) and name:
        return name.rsplit("/", 1)[-1] or None
    return None


def _no_attempt_error(ran: bool, reward: float | None, existing: str | None,
                      *, threshold: float = 1.0) -> str | None:
    """Stamp a silent 'the agent never ran' trial with a retryable ``NoAttempt`` error.

    An unresolved trial where the agent **never ran** — a repo clone / bootstrap failed in setup
    (e.g. deepswe's ``git bootstrap failed after 3 attempts``), so there's no native stream — is
    recorded by harbor as a bare ``reward=0`` with no error, so it reads as a genuine capability
    failure and hides from ``--retry-errors``. Surfacing it as ``NoAttempt`` makes it retryable.

    ``ran`` MUST be a robust activity signal — an actual parsed native stream — NOT harbor's
    ``agent_result`` token count, which is lossy (opencode on the harbor path reports 0 tokens even
    after a full 12-step run; keying off that produced FALSE no-attempts on genuine failures). No-op
    if an error is already set, the trial resolved, or the agent demonstrably ran."""
    if existing:
        return existing
    if reward is None:                 # ungraded — not our call (the retry-unresolved blind guard handles it)
        return None
    if reward >= threshold:            # resolved
        return None
    if not ran:
        return "NoAttempt: agent produced no attempt (no native stream — setup/clone failed)"
    return None


def _live_trial_task_id(tr: Any) -> str:
    """Recover a LIVE harbor ``TrialResult``'s full task id — its ``trial_name`` / dir slug is
    truncated to 32 chars, so prefer the recorded task path basename, then ``task_name``."""
    tid = getattr(tr, "task_id", None)
    path = getattr(tid, "path", None) if tid is not None else None
    if path:
        return Path(str(path)).name
    name = getattr(tr, "task_name", None)
    if isinstance(name, str) and name:
        return name.rsplit("/", 1)[-1]
    return getattr(tr, "trial_name", "") or ""


def _delete_trial_dirs(job_dir: Path, want: set[str]) -> int:
    """Remove the trial dirs whose RECORDED task id is in ``want`` so harbor's native resume re-runs
    exactly those — a trial that still has a ``result.json`` is otherwise matched-by-config and
    SKIPPED (only a *cancelled* trial is re-run natively). Truncation-safe: matches on the id the
    ``result.json`` records, not the 32-char-clipped dir name. Returns the count removed."""
    import json
    import shutil

    n = 0
    for rj in sorted(job_dir.glob("*/result.json")):
        try:
            tid = _harbor_trial_task_id(json.loads(rj.read_text()))
        except Exception:  # noqa: BLE001 — unreadable trial: fall back to the dir name
            tid = None
        if tid is None:
            name = rj.parent.name
            tid = name.rsplit("__", 1)[0] if "__" in name else name
        if tid in want:
            shutil.rmtree(rj.parent, ignore_errors=True)
            n += 1
    return n


def _result_from_harbor_json(d: dict[str, Any], task_id: str, trial_dir: Path,
                             *, resolved_threshold: float = 1.0) -> TaskResult:
    """Harbor's on-disk trial ``result.json`` → ``TaskResult`` (same fields
    :func:`_trial_to_result` reads from the live object, so resume round-trips)."""
    reward = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    ar = d.get("agent_result") or {}
    n_in, n_cache, n_out = (ar.get("n_input_tokens"), ar.get("n_cache_tokens"),
                            ar.get("n_output_tokens"))
    # Tokens + "did the agent run?" from the agent's OWN native stream (the canonical, agent-robust
    # source) — harbor's agent_result counters are lossy (opencode on the harbor path reports 0 even
    # after a full run). Fall back to harbor's counters only when there's no parseable stream.
    usage = usage_from_agent_dir(trial_dir / "agent")
    tokens = usage.to_token_counts() if usage else _tokens_from_harbor(n_in, n_cache, n_out)
    ran = usage is not None or ((n_in or 0) + (n_out or 0)) > 0 or (reward or 0) > 0
    # An agent-process crash (rc!=0) is recorded at agent_result.metadata.error, NOT exception_info —
    # fold it in so e.g. a monet "fetch failed" crash is errored, not a false reward=0. Then flag a
    # silent no-attempt (agent never ran — failed clone/bootstrap) so it's retryable, not a fake fail.
    error = _error_string(d.get("exception_info")) or _agent_error(ar)
    error = _no_attempt_error(ran, reward, error, threshold=resolved_threshold)
    return TaskResult(
        task_id=task_id,
        status=RolloutStatus.FAILED if error else RolloutStatus.COMPLETED,
        reward=reward,
        resolved=bool(reward is not None and reward >= resolved_threshold),
        error=error,
        tokens=tokens,
        artifact_dir=trial_dir,
        trajectory=TrajectoryRef(path=trial_dir, format="harbor-trial"),
    )


def _emit_trajectories(
    job_result: Any, items: list[tuple[Task, TaskContext]], agent_config: Any,
    *, run_dir: Path, job_name: str,
) -> None:
    """Convert each trial's native trajectory → ATIF ``agent/trajectory.json`` (best-effort).

    The generic shim path stashes the agent descriptor in ``agent_config.kwargs['identity']``
    (name/source-ref/model); the oracle/binding paths have none, so trajectories carry a
    blank agent and auto-detection simply no-ops when no known stream is present.
    """
    from beagle.benchmarks.trajectory import write_trajectory_json_auto

    identity = (getattr(agent_config, "kwargs", None) or {}).get("identity") or {}
    agent_name = identity.get("agent") or ""
    version = str((identity.get("source") or {}).get("ref") or "")
    model = identity.get("model")
    instr = {t.task_id: t.problem_statement for t, _ in items}
    for tr in getattr(job_result, "trial_results", []):
        name = getattr(tr, "trial_name", "") or ""
        instruction = instr.get(name) or instr.get(name.rsplit("__", 1)[0]) or ""
        try:
            write_trajectory_json_auto(
                run_dir / job_name / name / "agent", instruction=instruction,
                agent_name=agent_name, agent_version=version, model_name=model)
        except Exception:  # noqa: BLE001 — trajectory emission never fails the job
            pass


def _agent_identity(agent: Any) -> dict[str, Any]:
    """Serialize an agent into the small descriptor the harbor shim rebuilds from.

    Only JSON-primitive fields (name + source + config + model) — harbor spreads
    these through ``AgentConfig.kwargs``. The agent needs no harbor-specific code;
    this is what makes agent×harness integration additive (M + N), not M × N.
    """
    from beagle.agents.core.base import Evolvable

    source = None
    if isinstance(agent, Evolvable):
        s = agent.source()  # baseline or bound candidate; raises if repo unset
        source = {"repo": s.repo, "ref": s.ref, "entrypoint": s.entrypoint}
    spec = agent.spec
    return {
        "agent": spec.name,
        "source": source,
        "config": dict(spec.config or {}),
        "model": spec.model.name if spec.model else None,
    }


def _trial_to_result(
    trial: Any, *, jobs_dir: Path, job_name: str, resolved_threshold: float = 1.0
) -> TaskResult:
    """Map a harbor ``TrialResult`` → beagle ``TaskResult``, referencing native artifacts."""
    reward = None
    vr = getattr(trial, "verifier_result", None)
    if vr is not None and getattr(vr, "rewards", None):
        reward = vr.rewards.get("reward")
    error = (_error_string(getattr(trial, "exception_info", None))
             or _agent_error(getattr(trial, "agent_result", None)))
    trial_dir = jobs_dir / job_name / trial.trial_name
    try:
        n_in, n_cache, n_out, _cost = trial.compute_token_cost_totals()
    except Exception:  # noqa: BLE001 - token accounting is best-effort
        n_in = n_cache = n_out = None
    # Canonical usage + "did the agent run?" from the native stream (harbor's counters are lossy).
    usage = usage_from_agent_dir(trial_dir / "agent")
    tokens = usage.to_token_counts() if usage else _tokens_from_harbor(n_in, n_cache, n_out)
    ran = usage is not None or ((n_in or 0) + (n_out or 0)) > 0 or (reward or 0) > 0
    # A silent no-attempt (agent never ran — failed clone/bootstrap) → retryable NoAttempt, not fake 0.
    error = _no_attempt_error(ran, reward, error, threshold=resolved_threshold)
    return TaskResult(
        task_id=trial.trial_name,
        status=RolloutStatus.FAILED if error else RolloutStatus.COMPLETED,
        reward=reward,
        resolved=bool(reward is not None and reward >= resolved_threshold),
        error=error,
        tokens=tokens,
        artifact_dir=trial_dir,
        # The native harbor trial dir *is* the trajectory — it already contains
        # ``agent/`` (incl. the agent's own stream, e.g. monet.stream.jsonl),
        # ``verifier/``, and ``result.json``. We deliberately don't re-surface a
        # house per-agent trajectory: harbor's reward is authoritative and the tree
        # is byte-compatible with upstream (honor-the-native-harness).
        trajectory=TrajectoryRef(path=trial_dir, format="harbor-trial"),
    )


class PierHarness(HarborHarness):
    """Run tasks through **pier**'s Job driver — the harness for DeepSWE.

    Pier (``datacurve-pier``) is a harbor fork with an identical Job/config API, so this is just
    :class:`HarborHarness` retargeted at the ``pier`` package, pier's cluster environment, and a
    pier-native agent shim. The task source (:class:`~beagle.benchmarks.source.HarborCache` reads
    pier ``task.toml`` + ``instruction.md`` the same way) and the native
    ``<job>/<trial>/{agent,verifier,result.json}`` layout are shared, so resume
    (:meth:`completed`), trajectory emission, and result mapping are all inherited unchanged.
    Separate-verifier tasks (DeepSWE's ``environment_mode="separate"``) are handled by pier + the
    xrlenv pier adapter, not here.
    """

    FRAMEWORK = "pier"
    ENV_IMPORT_PATH = "xrlenv_plugins.pier:XrlenvPierEnvironmentCluster"
    SHIM_IMPORT_PATH = "beagle.benchmarks.harness._pier_agent:BeaglePierAgent"


class DockerHarness(BenchmarkHarness):
    """swebench-style: agent produces a patch, upstream evaluator grades it.

    Per-task: the agent's :class:`GenericBinding` runs the agent in an
    ``xrlenv.from_env()`` container to produce a unified diff; the base
    :meth:`BenchmarkHarness.rollout` loops :meth:`run`. Grading is deferred to the
    benchmark's :class:`~beagle.benchmarks.base.Grader` (a ``PatchEvalGrader`` that
    invokes the upstream evaluator natively).
    """

    def run(
        self,
        binding: RolloutBinding,
        task: Task,
        task_ctx: TaskContext,
        *,
        runtime: ContainerRuntime,
    ) -> TaskResult:
        if not isinstance(binding, GenericBinding):
            raise TypeError(f"DockerHarness requires a generic binding, got {binding.kind!r}")
        # The agent's own ``run`` is the single integration point: it acquires a container from
        # ``task_ctx.image`` via the (Runner-scoped) ``runtime``, runs the agent against the repo,
        # and returns a :class:`TaskResult` carrying the unified diff (``patch``). Grading is
        # deferred to the benchmark's ``PatchEvalGrader`` (which runs the upstream evaluator on
        # that patch) — nothing swebench-specific lives here.
        return binding.run(task, task_ctx, runtime=runtime)

    def completed(
        self, items: list[tuple[Task, TaskContext]], *, run_dir: Path
    ) -> list[TaskResult]:
        """Resume seam: read back the per-task ``result.json`` this harness's rollout+grade wrote
        under ``run_dir/<benchmark>/<task_id>/`` (the graded record — reward is set only after
        grading, so a re-readable ``result.json`` means the task is fully done). A task with no
        (or malformed) ``result.json`` simply re-runs."""
        from beagle.benchmarks.base import read_result_json

        out: list[TaskResult] = []
        for task, _ctx in items:
            rj = Path(run_dir) / (task.benchmark or "") / task.task_id / "result.json"
            r = read_result_json(rj) if rj.exists() else None
            if r is not None:
                out.append(r)
        return out


class NativeRunnerHarness(BenchmarkHarness):
    """The benchmark brings its *own* xrlenv-aware orchestrator; beagle invokes it.

    The strongest form of the drop-in contract: for a self-contained benchmark that
    already speaks xrlenv (WebArena-Infinity's ``run_eval_parallel_xrlenv.py``, and
    likely evoclaw), we **vendor the benchmark repo** and call its native runner —
    reusing its provisioning + verifier + answer-free lifecycle rather than
    reimplementing them.

    Agent injection stays **generic**: at the agent step the native runner runs the
    beagle agent via ``agent.run(task, task_ctx, runtime=<scoped to the provisioned
    container>)`` — the benchmark provides the environment, the agent does the acting,
    so an arbitrary agent works with no per-agent code in the benchmark and no fork.
    Evolving a source (e.g. monet) flows the candidate ref into the runner's install
    step. (The agent↔environment compatibility check — does an agent operate a browser
    vs a shell — is deferred to when this shape is built for real; see roadmap M3.)

    Subclasses override :meth:`rollout` to build the native-runner invocation and map
    its native results back to :class:`TaskResult`.
    """

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
        # TODO(subclass): invoke the vendored native runner for `items` (it owns
        # container reuse + app provisioning + verifier); at the agent step call
        # agent.run(...) in the provisioned container; read native results. Honor
        # `retry` (infra) + `attempt` (content-retry round) the way the runner it wraps does.
        raise NotImplementedError("NativeRunnerHarness subclass must implement rollout()")


__all__ = ["HarborHarness", "PierHarness", "DockerHarness", "NativeRunnerHarness"]
