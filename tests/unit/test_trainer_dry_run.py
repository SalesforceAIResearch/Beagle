"""The direct (PyTorch-UX) Trainer path: dataset carries its benchmark spec, the Trainer derives
the eval RunConfig from live objects, and `dry_run` resolves + prints the plan with no spend."""

from __future__ import annotations

import beagle as bgl
from beagle.agents.core.spec import AgentSource, AgentSpec, ModelSpec
from beagle.benchmarks.base import BenchmarkSpec
from beagle.data import TaskDataset


def _trainer(*, runtime="local") -> bgl.Trainer:
    evolvee = bgl.agents.build(AgentSpec(
        name="monet", model=ModelSpec(name="claude-opus-4-8"),
        source=AgentSource(repo="https://example.test/exp", ref="develop")))
    evolver = bgl.agents.build(AgentSpec(name="cursor", model=ModelSpec(name="auto")))
    algo = bgl.algorithms.build("darwinx", repo_root="/tmp/rr", evolvee_checkout="/tmp/co")
    return bgl.Trainer(evolvee=evolvee, evolver=evolver, algorithm=algo,
                      trainer_config={"runtime": {"kind": runtime}})


def _ds(task_ids=("t1",)) -> TaskDataset:
    spec = BenchmarkSpec(name="terminal_bench_2_1", task_ids=list(task_ids))
    return TaskDataset([], name=spec.name, benchmark_spec=spec)


def test_from_benchmark_stashes_the_spec(monkeypatch) -> None:
    monkeypatch.setattr("beagle.benchmarks.registry.load_tasks", lambda spec: iter([]))
    spec = BenchmarkSpec(name="b", task_ids=["t"])
    ds = TaskDataset.from_benchmark(spec)
    assert ds.benchmark_spec is spec and ds.name == "b"


def test_run_config_derived_from_dataset_and_evolvee() -> None:
    cfg = _trainer(runtime="xrlenv-cluster")._run_config(_ds(["t1", "t2"]))
    assert cfg is not None
    assert cfg.benchmark.name == "terminal_bench_2_1" and cfg.benchmark.task_ids == ["t1", "t2"]
    assert cfg.model.name == "claude-opus-4-8"                       # from the evolvee
    assert cfg.agent.source.repo == "https://example.test/exp"      # evolvee θ repo/ref
    assert cfg.agent.source.ref == "develop"
    assert cfg.runtime.kind == "xrlenv-cluster"                     # from trainer_config


def test_run_config_none_without_a_benchmark() -> None:
    plain = TaskDataset([], name="ad-hoc")   # no benchmark_spec
    assert _trainer()._run_config(plain) is None


def test_dry_run_prints_plan_and_returns_config(capsys) -> None:
    got = _trainer().dry_run(train_dataset=_ds(["adaptive-rejection-sampler"]))
    out = capsys.readouterr().out
    assert "dry-run" in out and "no spend" in out
    assert "monet @ https://example.test/exp#develop" in out
    assert "cursor" in out and "darwinx" in out
    assert "adaptive-rejection-sampler" in out
    assert got is not None and got.benchmark.task_ids == ["adaptive-rejection-sampler"]


def test_dry_run_reports_when_no_benchmark(capsys) -> None:
    got = _trainer().dry_run(train_dataset=TaskDataset([], name="ad-hoc"))
    assert got is None
    assert "none resolved" in capsys.readouterr().out
