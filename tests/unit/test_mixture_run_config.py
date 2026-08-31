"""Tests for scoring a mixture: RunConfig.benchmarks and the dataset provenance behind it.

Before this, a mixture was unscoreable and the reason was invisible: DataMixture produced a
correct interleaved dataset whose ``benchmark_spec`` was None (rightly -- it has several), the
Trainer derived the eval config from exactly that field, and the algorithm then refused to
score anything for want of a benchmark.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from beagle.benchmarks.base import BenchmarkSpec
from beagle.config import AgentConfig, BenchmarkConfig, ModelConfig, RunConfig
from beagle.data.dataset import TaskDataset
from beagle.data.mixture import DataMixture, MixtureComponent
from beagle.types import Task, TaskContext


def _items(benchmark: str, n: int):
    return [
        (Task(task_id=f"{benchmark}-{i}", benchmark=benchmark),
         TaskContext(image=None, benchmark_name=benchmark))
        for i in range(n)
    ]


def _ds(benchmark: str, n: int) -> TaskDataset:
    spec = BenchmarkSpec(name=benchmark)
    return TaskDataset(_items(benchmark, n), name=benchmark,
                       benchmark_spec=spec, benchmark_specs=[spec])


def _run_config(**kw) -> RunConfig:
    base = dict(
        model=ModelConfig(name="gpt-5.6-sol"),
        agent=AgentConfig(name="monet"),
        benchmark=BenchmarkConfig(name="swe-bench-verified"),
    )
    base.update(kw)
    return RunConfig(**base)


# -- RunConfig.benchmarks --------------------------------------------------------------


def test_single_benchmark_run_is_unchanged():
    cfg = _run_config()
    assert cfg.benchmarks is None
    assert [b.name for b in cfg.all_benchmarks()] == ["swe-bench-verified"]
    assert not cfg.is_mixture()


def test_mixture_lists_every_benchmark():
    cfg = _run_config(benchmarks=[BenchmarkConfig(name="swe-bench-verified"),
                                  BenchmarkConfig(name="deep-swe")])
    assert cfg.is_mixture()
    assert [b.name for b in cfg.all_benchmarks()] == ["swe-bench-verified", "deep-swe"]


def test_primary_must_lead_the_mixture():
    # Otherwise a mixture-unaware reader of .benchmark scores something that is not being run.
    with pytest.raises(ValidationError, match="must be the first entry"):
        _run_config(benchmarks=[BenchmarkConfig(name="deep-swe"),
                                BenchmarkConfig(name="swe-bench-verified")])


def test_empty_mixture_is_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        _run_config(benchmarks=[])


def test_duplicate_benchmarks_are_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        _run_config(benchmarks=[BenchmarkConfig(name="swe-bench-verified"),
                                BenchmarkConfig(name="swe-bench-verified")])


def test_for_benchmark_narrows_to_one_member():
    cfg = _run_config(benchmarks=[BenchmarkConfig(name="swe-bench-verified"),
                                  BenchmarkConfig(name="deep-swe")],
                      parallelism=8)
    narrowed = cfg.for_benchmark("deep-swe")
    assert narrowed.benchmark.name == "deep-swe"
    assert not narrowed.is_mixture()
    assert narrowed.parallelism == 8       # everything else survives the narrowing
    assert cfg.is_mixture()                # original untouched


def test_for_benchmark_rejects_a_non_member():
    with pytest.raises(KeyError, match="not part of this run"):
        _run_config().for_benchmark("deep-swe")


# -- dataset provenance ----------------------------------------------------------------


def test_from_a_single_benchmark_the_plural_has_one_entry():
    ds = _ds("swe-bench-verified", 3)
    assert [s.name for s in ds.benchmark_specs] == ["swe-bench-verified"]


def test_mixture_records_components_and_has_no_single_primary():
    mix = DataMixture([MixtureComponent(_ds("swe-bench-verified", 4)),
                       MixtureComponent(_ds("deep-swe", 4))])
    ds = mix.materialize()
    assert ds.benchmark_spec is None
    assert sorted(s.name for s in ds.benchmark_specs) == ["deep-swe", "swe-bench-verified"]


def test_split_carries_provenance_to_both_halves():
    # The val half is the one that gets scored by the held-out veto; losing its specs here
    # would make it silently unscoreable.
    mix = DataMixture([MixtureComponent(_ds("swe-bench-verified", 6)),
                       MixtureComponent(_ds("deep-swe", 4))], val_fraction=0.2)
    train, val = mix.split()
    for half in (train, val):
        assert sorted(s.name for s in half.benchmark_specs) == ["deep-swe", "swe-bench-verified"]


def test_filter_and_select_keep_provenance():
    ds = _ds("swe-bench-verified", 4)
    assert ds.filter(lambda t: True).benchmark_specs == ds.benchmark_specs
    assert ds.select(["swe-bench-verified-0"]).benchmark_specs == ds.benchmark_specs


def test_concat_merges_provenance_and_drops_a_mismatched_primary():
    joined = _ds("swe-bench-verified", 2).concat(_ds("deep-swe", 2))
    assert sorted(s.name for s in joined.benchmark_specs) == ["deep-swe", "swe-bench-verified"]
    assert joined.benchmark_spec is None


def test_concat_of_the_same_benchmark_keeps_its_primary():
    a, b = _ds("swe-bench-verified", 2), _ds("swe-bench-verified", 2)
    joined = a.concat(b)
    assert joined.benchmark_spec is not None
    assert [s.name for s in joined.benchmark_specs] == ["swe-bench-verified"]
