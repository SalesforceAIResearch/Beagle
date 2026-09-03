"""``beagle.eval`` — evaluate an agent on a benchmark (or a materialized mix) → ``run.json``.

The faithful beagle mapping of a ``runner.run``-style entry point: build the agent + dataset
from a :class:`~beagle.config.RunConfig`, roll every task out through its benchmark's *native*
harness (harbor.Job for terminal-bench), grade, and write the canonical ``run.json``. This is
the framework's single evaluation seam — ``beagle evaluate`` wraps it, native evolution algorithms
call it, and the vendored-DarwinX adapter (:mod:`beagle.algorithms.darwinx.eval`) goes through
it before reshaping the output. It does one thing: **eval a given agent on a selected (mix of)
benchmark(s)** — nothing evolution-specific lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from beagle.config import RunConfig
    from beagle.rollout.runner import RunResult


def evaluate(
    config: RunConfig,
    *,
    results_root: str | Path | None = None,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    resume: bool = False,
    retry_errors: bool = False,
    retry_unresolved: bool = False,
    only_task_ids: set[str] | None = None,
    force_resume: bool = False,
    campaign_id: str | None = None,
    config_path: str | Path | None = None,
    agent: Any = None,
    dataset: Any = None,
    runtime: Any = None,
) -> RunResult:
    """Evaluate one agent on its benchmark(s) → a run dir with ``run.json``; return the
    :class:`~beagle.rollout.runner.RunResult`.

    The agent + dataset are built from ``config`` unless supplied. Pass a
    ``DataMixture``-materialized ``dataset`` to evaluate a **mix** of benchmarks in one run —
    the Runner groups by ``Task.benchmark`` and writes a benchmark-keyed ``run.json``.
    ``run_dir`` sets the output dir verbatim; otherwise ``results_root/<run_id>``.

    ``runtime`` is the container substrate the Runner hands to each task's harness. Leave it
    ``None`` for the harbor path (harbor owns the trial environment); pass a
    :class:`~beagle.rollout.runtime.ContainerRuntime` (e.g. from ``build_runtime``) when the
    mix includes a container-harness benchmark such as SWE-bench, whose ``DockerHarness``
    acquires the per-instance image through it.
    """
    import beagle as bgl
    from beagle.rollout.runner import Runner

    if agent is None:
        agent = bgl.agents.build(config.agent_spec())
    if dataset is None:
        specs = config.all_benchmark_specs()
        dataset = bgl.TaskDataset.from_benchmark(specs[0])
        for spec in specs[1:]:          # a mixture: concat keeps each benchmark's own selection
            dataset = dataset.concat(bgl.TaskDataset.from_benchmark(spec))
    return Runner(runtime, parallelism=config.parallelism,
                  eval_parallelism=config.parallelism_eval_patches, results_root=results_root).run(
        agent, dataset, config=config, run_id=run_id, run_dir=run_dir,
        resume=resume, retry_errors=retry_errors, retry_unresolved=retry_unresolved,
        only_task_ids=only_task_ids, force_resume=force_resume, config_path=config_path,
        campaign_id=campaign_id,
    )


__all__ = ["evaluate"]
