"""Trainer — the thin combiner: turn data into an `evaluate` callable and hand off to
`algorithm.evolve`. Hermetic: fake algorithms capture what the Trainer passes; the Runner is
faked for the evaluate-closure test. No loop lives in the Trainer."""

from __future__ import annotations

from types import SimpleNamespace

import beagle as bgl
from beagle.agents.core.spec import AgentSource
from beagle.algorithms.base import Candidate, EvolveAlgorithm


class _RecordingAlgo(EvolveAlgorithm):
    def evolve(self, *, evaluate, evolvee, evolver, val=None, config=None):  # noqa: ANN001
        self.seen = dict(evolvee=evolvee, evolver=evolver, has_val=val is not None,
                         config=config, evaluate_callable=callable(evaluate))
        return Candidate(id="best", source=AgentSource(repo="r", ref="x"))


def test_fit_hands_data_and_agents_to_algorithm_evolve() -> None:
    evolvee, evolver = bgl.agents.build("monet"), bgl.agents.build("cursor")
    algo = _RecordingAlgo()
    best = bgl.Trainer(evolvee=evolvee, evolver=evolver, algorithm=algo).fit(
        train_dataset=bgl.TaskDataset([]))

    assert best is not None and best.id == "best"
    assert algo.seen["evolvee"] is evolvee and algo.seen["evolver"] is evolver
    assert algo.seen["has_val"] is False and algo.seen["evaluate_callable"] is True


def test_fit_evaluate_closure_binds_candidate_source_and_runs(monkeypatch) -> None:
    """The `evaluate` the Trainer builds: rebind the evolvee to the candidate's source, roll
    out on the data via the Runner (with the per-candidate RunConfig), record fitness."""
    from beagle.config import BeagleConfig

    captured: dict = {}

    class _FakeRunner:
        def __init__(self, **kw):
            pass

        def run(self, agent, dataset, *, config):  # noqa: ANN001
            captured["agent_ref"] = agent.source().ref
            captured["benchmark"] = config.benchmark.name       # config threaded from the evolution cfg
            return SimpleNamespace(score=0.5, results=[1, 2])

    class _EvalAlgo(EvolveAlgorithm):
        def evolve(self, *, evaluate, evolvee, evolver, val=None, config=None):  # noqa: ANN001
            c = Candidate(id="c", source=AgentSource(repo="r", ref="cand-branch"))
            evaluate(c)
            return c

    monkeypatch.setattr("beagle.trainer.Runner", _FakeRunner)
    cfg = BeagleConfig.from_dict({
        "evolvee": {"name": "monet", "model": {"name": "gpt-5.5"}, "config": {}},
        "evolver": {"name": "cursor", "config": {}},
        "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t1"]},
    })
    t = bgl.Trainer.from_config(cfg)
    t.algorithm = _EvalAlgo()                                    # swap in the eval-invoking algo
    best = t.fit(train_dataset=bgl.TaskDataset([]))

    assert best is not None and best.score == 0.5 and best.results == [1, 2]
    assert captured["agent_ref"] == "cand-branch"               # evolvee rebound to the candidate
    assert captured["benchmark"] == "terminal_bench_2_1"


def test_from_config_assembles_agents_algorithm_and_keeps_config() -> None:
    from beagle.config import BeagleConfig

    cfg = BeagleConfig.from_dict({
        "evolvee": {"name": "monet", "model": {"name": "gpt-5.5"}, "config": {}},
        "evolver": {"name": "cursor", "config": {}},
        "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t1"]},
        "algorithm": {"name": "darwinx", "hparams": {"max_loop_iters": 4}},
    })
    t = bgl.Trainer.from_config(cfg)

    assert t.evolvee.name == "monet" and t.evolver.name == "cursor"
    assert type(t.algorithm).__name__ == "DarwinX" and t.algorithm.hparams["max_loop_iters"] == 4
    # kept the BeagleConfig → fit can derive the per-candidate eval RunConfig
    rc = t._run_config()
    assert rc is not None and rc.benchmark.name == "terminal_bench_2_1" and rc.model.name == "gpt-5.5"
