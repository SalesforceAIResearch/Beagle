"""`build` accepts the declarative Config layer — not just names / internal specs. This is the
design surface the PyTorch-UX example uses: AgentConfig, DarwinXConfig, BenchmarkConfig."""

from __future__ import annotations

import dataclasses

import pytest

import beagle as bgl
from beagle.algorithms import AlgorithmConfig, DarwinX, DarwinXConfig
from beagle.config import AgentConfig, AgentSourceConfig, BenchmarkConfig, ModelConfig
from beagle.data import TaskDataset


def test_agents_build_accepts_agent_config() -> None:
    ev = bgl.agents.build(AgentConfig(name="cursor", model=ModelConfig(name="auto")))
    assert ev.name == "cursor" and ev.spec.model is not None and ev.spec.model.name == "auto"


def test_agents_build_accepts_agent_config_with_source() -> None:
    a = bgl.agents.build(AgentConfig(name="monet", model=ModelConfig(name="m"),
                                    source=AgentSourceConfig(repo="r", ref="x")))
    assert a.spec.source is not None and a.spec.source.repo == "r" and a.spec.source.ref == "x"


def test_agents_build_rejects_overrides_with_a_config() -> None:
    with pytest.raises(TypeError, match="AgentConfig"):
        bgl.agents.build(AgentConfig(name="cursor"), role="evolver")


def test_algorithms_build_from_config_instance_infers_algorithm() -> None:
    algo = bgl.algorithms.build(DarwinXConfig(repo_root="/tmp/rr", max_loop_iters=1))
    assert isinstance(algo, DarwinX)
    assert algo.config.repo_root == "/tmp/rr" and algo.config.max_loop_iters == 1


def test_algorithms_build_unknown_config_type_raises() -> None:
    class _Stray(AlgorithmConfig):
        pass

    with pytest.raises(KeyError, match="no registered algorithm"):
        bgl.algorithms.build(_Stray())


def test_from_benchmark_accepts_benchmark_config(monkeypatch) -> None:
    monkeypatch.setattr("beagle.benchmarks.registry.load_tasks", lambda spec: iter([]))
    ds = TaskDataset.from_benchmark(BenchmarkConfig(name="b", task_ids=["t"]))
    # the config is resolved to its BenchmarkSpec (a dataclass) and kept for RunConfig derivation
    assert dataclasses.is_dataclass(ds.benchmark_spec)
    assert ds.benchmark_spec.name == "b" and ds.benchmark_spec.task_ids == ["t"]
