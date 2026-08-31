"""``beagle.eval.evaluate`` — the general "eval an agent on a benchmark" seam (the faithful
runner.run mapping). Hermetic: the Runner is faked; we assert evaluate builds nothing extra
when agent/dataset are supplied and threads every knob into Runner.run."""

from __future__ import annotations

from types import SimpleNamespace

import beagle as bgl
from beagle.config import RunConfig


def _cfg() -> RunConfig:
    return RunConfig.from_dict({
        "model": {"name": "gpt-5.5"}, "agent": {"name": "monet", "config": {}},
        "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t1"]}, "parallelism": 3,
    })


def test_evaluate_threads_everything_into_the_runner(monkeypatch) -> None:
    seen: dict = {}

    class _FakeRunner:
        def __init__(self, runtime=None, *, parallelism, eval_parallelism=None, results_root):  # noqa: ANN001
            seen["runtime"] = runtime
            seen["parallelism"], seen["results_root"] = parallelism, results_root
            seen["eval_parallelism"] = eval_parallelism

        def run(self, agent, dataset, *, config, run_id, run_dir, resume, retry_errors,
                retry_unresolved, only_task_ids, force_resume, config_path, campaign_id):  # noqa: ANN001
            seen.update(agent=agent, dataset=dataset, run_id=run_id, run_dir=run_dir,
                        resume=resume, retry=retry_errors, retry_unresolved=retry_unresolved,
                        only_task_ids=only_task_ids, force_resume=force_resume,
                        config_path=config_path, campaign=campaign_id)
            return SimpleNamespace(run_id=run_id or "auto")

    monkeypatch.setattr("beagle.rollout.runner.Runner", _FakeRunner)

    agent, dataset, runtime = object(), [("t1", None)], object()
    out = bgl.evaluate(_cfg(), results_root="R", run_id="RID", run_dir="D", resume=True,
                      retry_errors=True, config_path="c.yaml", campaign_id="camp",
                      agent=agent, dataset=dataset, runtime=runtime)

    assert out.run_id == "RID"
    assert seen == {
        "runtime": runtime,  # the container substrate, handed to the Runner (None on the harbor path)
        "parallelism": 3, "eval_parallelism": None, "results_root": "R", "agent": agent, "dataset": dataset,
        "run_id": "RID", "run_dir": "D", "resume": True, "retry": True, "retry_unresolved": False,
        "only_task_ids": None, "force_resume": False,
        "config_path": "c.yaml", "campaign": "camp",
    }  # supplied agent/dataset used verbatim (nothing rebuilt); every knob threaded through
