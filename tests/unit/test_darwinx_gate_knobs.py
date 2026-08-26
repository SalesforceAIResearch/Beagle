"""The gate knobs a hosted multi-benchmark campaign needs, and the coupling it must not get wrong.

Two things are pinned here.

The per-benchmark sample-size convention is *mirrored* in DarwinXConfig rather than imported from the
vendored driver, so that a hosted run does not depend on a private helper in vendored code. A mirror
is a second source of truth, and the only thing that makes one safe is a test that fails when the two
disagree: a silently wrong suffix produces no error, just a benchmark quietly running at the global
default sample size.

The other is the defer/panel coupling, which is a load-time error precisely because nothing about the
run looks wrong while it happens -- node scores rise.
"""

import pytest

from beagle.algorithms.darwinx.config import DarwinXConfig
from beagle.algorithms.darwinx.vendor.evolve.multibench import _bench_env_suffix


@pytest.mark.parametrize("benchmark", [
    "deep-swe",
    "swe-bench-pro",
    "harbor-swebench-verified",
    "swe-bench-lite",
])
def test_the_per_benchmark_suffix_matches_the_driver(benchmark):
    """The config's mirrored suffix rule must agree with the driver's, name for name."""
    cfg = DarwinXConfig(mixture_gate_screen_tasks_per_benchmark={benchmark: 7})
    expected = f"DARWINX_GATE_MIXTURE_GATE_SCREEN_TASKS_{_bench_env_suffix(benchmark)}"
    assert cfg.to_driver_env().get(expected) == "7", cfg.to_driver_env()


def test_both_sizes_can_be_overridden_per_benchmark():
    cfg = DarwinXConfig(
        mixture_gate_tasks=80,
        mixture_gate_screen_tasks=25,
        mixture_gate_tasks_per_benchmark={"deep-swe": 24},
        mixture_gate_screen_tasks_per_benchmark={"deep-swe": 8},
    )
    env = cfg.to_driver_env()
    # The global sizes still stand for the benchmarks that did not ask for anything else.
    assert env["DARWINX_GATE_MIXTURE_GATE_TASKS"] == "80"
    assert env["DARWINX_GATE_MIXTURE_GATE_SCREEN_TASKS"] == "25"
    assert env["DARWINX_GATE_MIXTURE_GATE_TASKS_DEEP_SWE"] == "24"
    assert env["DARWINX_GATE_MIXTURE_GATE_SCREEN_TASKS_DEEP_SWE"] == "8"


def test_unset_knobs_emit_nothing():
    """The driver's own defaults have to stand for anything the campaign did not set."""
    env = DarwinXConfig().to_driver_env()
    assert not [k for k in env if k.startswith("DARWINX_GATE_MIXTURE_GATE")]
    assert "DARWINX_GATE_FIXED_EVAL_PANEL" not in env
    assert "DARWINX_GATE_EVAL_PANEL_SIZE" not in env


def test_defer_without_a_shared_panel_is_a_load_time_error():
    with pytest.raises(ValueError, match="fixed_eval_panel"):
        DarwinXConfig(defer_node_full_eval=True)


def test_defer_with_a_panel_is_fine():
    cfg = DarwinXConfig(defer_node_full_eval=True, fixed_eval_panel=True, eval_panel_size=40)
    env = cfg.to_driver_env()
    assert env["DARWINX_GATE_DEFER_NODE_FULL_EVAL"] == "1"
    assert env["DARWINX_GATE_FIXED_EVAL_PANEL"] == "1"
    assert env["DARWINX_GATE_EVAL_PANEL_SIZE"] == "40"


def test_a_shared_panel_without_defer_is_allowed():
    """Only one direction is a trap: a panel with the full per-node eval is merely expensive."""
    assert DarwinXConfig(fixed_eval_panel=True).to_driver_env()["DARWINX_GATE_FIXED_EVAL_PANEL"] == "1"


def test_a_negative_panel_size_is_rejected():
    with pytest.raises(ValueError, match="eval_panel_size"):
        DarwinXConfig(fixed_eval_panel=True, eval_panel_size=-1)


def test_panel_size_zero_means_the_whole_subset():
    """0 is meaningful, not missing, so it has to survive to the driver."""
    cfg = DarwinXConfig(defer_node_full_eval=True, fixed_eval_panel=True, eval_panel_size=0)
    assert cfg.to_driver_env()["DARWINX_GATE_EVAL_PANEL_SIZE"] == "0"
