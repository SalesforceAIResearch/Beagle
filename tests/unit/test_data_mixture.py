"""Tests for DataMixture and the weighted round-robin sampler.

The multi-benchmark evolve runs on top of these three previously-stubbed pieces, so the
properties worth pinning are the ones a training mixture silently depends on: no task is
ever dropped, the interleave matches the requested ratio, and the same config twice
produces the same ordering.
"""

from __future__ import annotations

import pytest

from beagle.data.dataset import TaskDataset
from beagle.data.mixture import DataMixture, MixtureComponent
from beagle.data.sampler import ConcatSampler, WeightedRoundRobinSampler
from beagle.types import Task, TaskContext


def _items(benchmark: str, n: int):
    """n (Task, TaskContext) pairs tagged with a benchmark, ids like 'swev-0'."""
    return [
        (
            Task(task_id=f"{benchmark}-{i}", benchmark=benchmark),
            TaskContext(image=None, benchmark_name=benchmark),
        )
        for i in range(n)
    ]


def _benchmarks_of(items) -> list[str]:
    return [t.benchmark for t, _ in items]


# -- WeightedRoundRobinSampler ---------------------------------------------------------


def test_wrr_interleaves_in_the_requested_ratio():
    a, b = _items("a", 6), _items("b", 2)
    out = WeightedRoundRobinSampler().combine([a, b], [3.0, 1.0])
    assert _benchmarks_of(out) == ["a", "a", "a", "b", "a", "a", "a", "b"]


def test_wrr_preserves_every_item():
    a, b, c = _items("a", 7), _items("b", 3), _items("c", 5)
    out = WeightedRoundRobinSampler().combine([a, b, c], [3.0, 1.0, 2.0])
    assert len(out) == 15
    assert {t.task_id for t, _ in out} == {t.task_id for t, _ in a + b + c}


def test_wrr_keeps_each_group_in_its_original_order():
    a, b = _items("a", 4), _items("b", 4)
    out = WeightedRoundRobinSampler().combine([a, b], [2.0, 1.0])
    a_ids = [t.task_id for t, _ in out if t.benchmark == "a"]
    assert a_ids == [f"a-{i}" for i in range(4)]


def test_wrr_is_deterministic_across_equal_weights():
    # Equal weights make every group's draw times coincide, which is exactly the case
    # float arithmetic would break inconsistently.
    groups = [_items("a", 5), _items("b", 5), _items("c", 5)]
    first = _benchmarks_of(WeightedRoundRobinSampler().combine(groups, [1.0, 1.0, 1.0]))
    second = _benchmarks_of(WeightedRoundRobinSampler().combine(groups, [1.0, 1.0, 1.0]))
    assert first == second
    assert first[:3] == ["a", "b", "c"]  # ties resolve to the lowest index


def test_wrr_drains_the_remainder_once_a_group_is_exhausted():
    a, b = _items("a", 2), _items("b", 5)
    out = _benchmarks_of(WeightedRoundRobinSampler().combine([a, b], [3.0, 1.0]))
    assert out.count("a") == 2 and out.count("b") == 5
    assert out[-3:] == ["b", "b", "b"]  # a ran out; b closes ranks


def test_wrr_tolerates_empty_groups():
    out = WeightedRoundRobinSampler().combine([_items("a", 3), []], [1.0, 5.0])
    assert _benchmarks_of(out) == ["a", "a", "a"]


def test_wrr_rejects_nonpositive_weight_rather_than_dropping_tasks():
    with pytest.raises(ValueError, match="weights must be > 0"):
        WeightedRoundRobinSampler().combine([_items("a", 2), _items("b", 2)], [1.0, 0.0])


def test_wrr_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="must correspond"):
        WeightedRoundRobinSampler().combine([_items("a", 1)], [1.0, 2.0])


# -- DataMixture.materialize / split ---------------------------------------------------


def test_materialize_applies_limit_per_component():
    mix = DataMixture(
        components=[
            MixtureComponent(TaskDataset(_items("a", 10)), weight=1.0, limit=3),
            MixtureComponent(TaskDataset(_items("b", 10)), weight=1.0, limit=2),
        ],
        sampler=ConcatSampler(),
    )
    assert _benchmarks_of(materialized := mix.materialize()) == ["a"] * 3 + ["b"] * 2
    assert materialized.name == "mixture"


def test_split_holds_out_the_val_fraction():
    mix = DataMixture(
        components=[MixtureComponent(TaskDataset(_items("a", 10)))],
        val_fraction=0.2,
    )
    train, val = mix.split()
    assert (len(train), len(val)) == (8, 2)


def test_tasks_remember_their_benchmark_through_the_mix():
    # Per-benchmark scoring downstream depends entirely on this surviving the interleave.
    mix = DataMixture(
        components=[
            MixtureComponent(TaskDataset(_items("swev", 4)), weight=2.0),
            MixtureComponent(TaskDataset(_items("deepswe", 4)), weight=1.0),
        ],
        sampler=WeightedRoundRobinSampler(),
    )
    assert set(_benchmarks_of(mix.materialize())) == {"swev", "deepswe"}


# -- DataMixture.from_config -----------------------------------------------------------


def test_from_config_parses_components_and_sampler():
    mix = DataMixture.from_config(
        {
            "components": [
                {"benchmark": {"name": "swe-bench-verified"}, "weight": 3.0, "limit": 100},
                {"benchmark": {"name": "deep-swe"}, "weight": 1.0},
            ],
            "sampler": "weighted_round_robin",
            "val_fraction": 0.1,
        }
    )
    assert len(mix.components) == 2
    assert isinstance(mix.sampler, WeightedRoundRobinSampler)
    assert mix.val_fraction == pytest.approx(0.1)
    assert (mix.components[0].weight, mix.components[0].limit) == (3.0, 100)
    assert (mix.components[1].weight, mix.components[1].limit) == (1.0, None)
    assert mix.components[0].benchmark_spec.name == "swe-bench-verified"


def test_from_config_defaults_to_concat_and_no_holdout():
    mix = DataMixture.from_config({"components": [{"benchmark": {"name": "deep-swe"}}]})
    assert isinstance(mix.sampler, ConcatSampler)
    assert mix.val_fraction == 0.0
    assert mix.components[0].weight == 1.0


def test_from_config_passes_through_a_prebuilt_dataset():
    ds = TaskDataset(_items("a", 2))
    mix = DataMixture.from_config({"components": [{"benchmark": ds}]})
    assert mix.materialize().task_ids == ["a-0", "a-1"]


@pytest.mark.parametrize(
    "config, message",
    [
        ({"components": []}, "at least one"),
        ({"components": [{}]}, "no 'benchmark'"),
        ({"components": [{"benchmark": {"name": "x"}}], "sampler": "nope"}, "unknown sampler"),
    ],
)
def test_from_config_rejects_bad_input(config, message):
    with pytest.raises((ValueError, TypeError), match=message):
        DataMixture.from_config(config)
