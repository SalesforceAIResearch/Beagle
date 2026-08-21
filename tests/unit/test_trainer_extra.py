"""Additional coverage for the Trainer — None-config guard, val_dataset path,
from_config when benchmark is absent, and role-validation errors.

All tests are hermetic: the Runner is monkeypatched so no rollouts run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import beagle as bgl
from beagle.agents.core.spec import AgentSource
from beagle.algorithms.base import Candidate, EvolveAlgorithm
from beagle.trainer import Trainer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _EvalCapturingAlgo(EvolveAlgorithm):
    """Calls evaluate once on a fake candidate, records val presence."""

    def __init__(self, *, call_evaluate: bool = False, call_val: bool = False) -> None:
        super().__init__()
        self.call_evaluate = call_evaluate
        self.call_val = call_val
        self.saw_val: bool = False
        self.eval_error: Exception | None = None

    def evolve(self, *, evaluate, evolvee, evolver, val=None, config=None):  # noqa: ANN001
        self.saw_val = val is not None
        c = Candidate(id="c", source=AgentSource(repo="r", ref="ref"))
        if self.call_evaluate:
            try:
                evaluate(c)
            except Exception as exc:
                self.eval_error = exc
        if self.call_val and val is not None:
            val(c)
        return c


# ---------------------------------------------------------------------------
# _make_evaluate RuntimeError when config is None
# ---------------------------------------------------------------------------

def test_make_evaluate_raises_when_config_none() -> None:
    """When no benchmark is in the config, _make_evaluate builds a closure that
    raises RuntimeError on first call — the algorithm surfaces it, not the Trainer."""
    algo = _EvalCapturingAlgo(call_evaluate=True)
    trainer = Trainer(
        evolvee=bgl.agents.build("monet"),
        evolver=bgl.agents.build("cursor"),
        algorithm=algo,
    )
    # _xcfg is None → _run_config() returns None → evaluate raises
    trainer.fit(train_dataset=bgl.TaskDataset([]))

    assert isinstance(algo.eval_error, RuntimeError)
    assert "no `benchmark`" in str(algo.eval_error)


def test_make_evaluate_raises_when_xcfg_has_no_benchmark() -> None:
    """from_config with benchmark=None → _run_config()=None → same RuntimeError."""
    from beagle.config import BeagleConfig

    algo = _EvalCapturingAlgo(call_evaluate=True)
    cfg = BeagleConfig.from_dict({
        "evolvee": {"name": "monet", "model": {"name": "gpt-5.5"}, "config": {}},
        "evolver": {"name": "cursor", "config": {}},
        # No "benchmark" key → benchmark is None
    })
    t = Trainer.from_config(cfg)
    t.algorithm = algo
    t.fit(train_dataset=bgl.TaskDataset([]))

    assert isinstance(algo.eval_error, RuntimeError)
    assert "no `benchmark`" in str(algo.eval_error)


# ---------------------------------------------------------------------------
# val_dataset path
# ---------------------------------------------------------------------------

def test_fit_passes_val_callable_when_val_dataset_provided(monkeypatch) -> None:
    """When val_dataset is provided, Trainer.fit builds a val callable and
    passes it to algorithm.evolve."""
    from beagle.config import BeagleConfig

    class _FakeRunner:
        def __init__(self, **kw): pass
        def run(self, agent, dataset, *, config):  # noqa: ANN001
            return SimpleNamespace(score=1.0, results=[])

    monkeypatch.setattr("beagle.trainer.Runner", _FakeRunner)

    algo = _EvalCapturingAlgo()
    cfg = BeagleConfig.from_dict({
        "evolvee": {"name": "monet", "model": {"name": "gpt-5.5"}, "config": {}},
        "evolver": {"name": "cursor", "config": {}},
        "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t1"]},
    })
    t = Trainer.from_config(cfg)
    t.algorithm = algo
    t.fit(train_dataset=bgl.TaskDataset([]), val_dataset=bgl.TaskDataset([]))

    assert algo.saw_val is True  # val callable was threaded through


def test_fit_passes_no_val_callable_when_val_dataset_absent() -> None:
    """When val_dataset is omitted, the algorithm receives val=None."""
    algo = _EvalCapturingAlgo()
    Trainer(
        evolvee=bgl.agents.build("monet"),
        evolver=bgl.agents.build("cursor"),
        algorithm=algo,
    ).fit(train_dataset=bgl.TaskDataset([]))

    assert algo.saw_val is False


def test_val_evaluate_binds_source_and_runs(monkeypatch) -> None:
    """The val closure has the same behavior as the train closure — it rebinds
    the evolvee to the candidate's source and calls Runner.run."""
    from beagle.config import BeagleConfig

    captured: list[str] = []

    class _FakeRunner:
        def __init__(self, **kw): pass
        def run(self, agent, dataset, *, config):  # noqa: ANN001
            captured.append(agent.source().ref)
            return SimpleNamespace(score=0.8, results=[])

    monkeypatch.setattr("beagle.trainer.Runner", _FakeRunner)

    class _ValCallingAlgo(EvolveAlgorithm):
        def evolve(self, *, evaluate, evolvee, evolver, val=None, config=None):  # noqa: ANN001
            c = Candidate(id="c", source=AgentSource(repo="r", ref="val-branch"))
            val(c)  # type: ignore[misc]
            return c

    cfg = BeagleConfig.from_dict({
        "evolvee": {"name": "monet", "model": {"name": "gpt-5.5"}, "config": {}},
        "evolver": {"name": "cursor", "config": {}},
        "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t1"]},
    })
    t = Trainer.from_config(cfg)
    t.algorithm = _ValCallingAlgo()
    best = t.fit(train_dataset=bgl.TaskDataset([]), val_dataset=bgl.TaskDataset([]))

    assert best is not None and best.score == 0.8
    assert captured == ["val-branch"]  # evolvee was rebound to the candidate


# ---------------------------------------------------------------------------
# role validation
# ---------------------------------------------------------------------------

def test_trainer_rejects_non_evolvee_as_evolvee() -> None:
    """An agent that can't be an evolvee (no Runnable+Evolvable) raises TypeError."""
    with pytest.raises(TypeError, match="cannot be an evolvee"):
        Trainer(
            evolvee=bgl.agents.build("cursor"),   # cursor is Editor only, not Evolvable+Runnable
            evolver=bgl.agents.build("cursor"),
            algorithm=bgl.algorithms.build("darwinx"),
        )


def test_trainer_stamps_roles_on_agents() -> None:
    """Trainer.__init__ stamps EVOLVEE/EVOLVER roles onto the agent specs."""
    from beagle.types import AgentRole

    evolvee = bgl.agents.build("monet")
    evolver = bgl.agents.build("cursor")
    Trainer(evolvee=evolvee, evolver=evolver, algorithm=bgl.algorithms.build("darwinx"))

    assert evolvee.spec.role == AgentRole.EVOLVEE
    assert evolver.spec.role == AgentRole.EVOLVER


# ---------------------------------------------------------------------------
# from_config edge cases
# ---------------------------------------------------------------------------

def test_from_config_benchmark_none_run_config_is_none() -> None:
    """from_config with no benchmark → _run_config() is None, Trainer still
    builds without error."""
    from beagle.config import BeagleConfig

    cfg = BeagleConfig.from_dict({
        "evolvee": {"name": "monet", "model": {"name": "gpt-5.5"}, "config": {}},
        "evolver": {"name": "cursor", "config": {}},
    })
    t = Trainer.from_config(cfg)
    assert t._run_config() is None


def test_from_config_without_trainer_config_uses_empty_dict() -> None:
    """trainer_config=None defaults to an empty dict (no KeyError on config.get)."""
    from beagle.config import BeagleConfig

    cfg = BeagleConfig.from_dict({
        "evolvee": {"name": "monet", "model": {"name": "gpt-5.5"}, "config": {}},
        "evolver": {"name": "cursor", "config": {}},
    })
    t = Trainer.from_config(cfg, trainer_config=None)
    assert isinstance(t.config, dict)
