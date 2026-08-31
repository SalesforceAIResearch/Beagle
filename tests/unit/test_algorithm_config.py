"""Typed algorithm config: generic AlgorithmConfig + build interface, and DarwinX's subclass
(drift guard + gate→env translation)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import beagle as bgl
from beagle.algorithms import AlgorithmConfig, DarwinXConfig
from beagle.algorithms.base import EvolveAlgorithm


# --- generic surface ---------------------------------------------------------

def test_base_config_is_permissive_and_has_empty_driver_env() -> None:
    # The base accepts arbitrary knobs (a trivial algorithm needs no subclass) and exposes none.
    cfg = AlgorithmConfig(anything=1)
    assert cfg.to_driver_env() == {}


def test_build_validates_kwargs_against_the_algorithm_config() -> None:
    algo = bgl.algorithms.build("darwinx", repo_root="/tmp/rr", max_loop_iters=1)
    assert isinstance(algo.config, DarwinXConfig)
    assert algo.config.repo_root == "/tmp/rr" and algo.config.max_loop_iters == 1


def test_build_accepts_a_ready_config_object() -> None:
    cfg = DarwinXConfig(repo_root="/tmp/x", campaign="c")
    algo = bgl.algorithms.build("darwinx", config=cfg)
    assert algo.config is cfg


def test_from_config_default() -> None:
    cfg = DarwinXConfig(repo_root="/tmp/y")
    assert bgl.algorithms.DarwinX.from_config(cfg).config is cfg


def test_hparams_shows_only_set_knobs() -> None:
    algo = bgl.algorithms.build("darwinx", repo_root="/tmp/rr", max_loop_iters=8)
    assert algo.hparams == {"repo_root": "/tmp/rr", "max_loop_iters": 8}   # defaults excluded


# --- DarwinX drift guard + env translation -----------------------------------

def test_unknown_knob_fails_loud() -> None:
    with pytest.raises(ValidationError):
        bgl.algorithms.build("darwinx", repo_rooot="typo")   # misspelled → drift guard


def test_to_driver_env_translates_only_set_gate_knobs() -> None:
    cfg = DarwinXConfig(cross_bench_gate=True, equivalence_gate=False, fitness_alpha=0.3,
                        verifier_model="gpt-5.5")
    assert cfg.to_driver_env() == {
        "DARWINX_GATE_CROSS_BENCH_GATE": "1",
        "DARWINX_GATE_EQUIVALENCE_GATE_ENABLED": "0",   # explicit False → "0"
        "DARWINX_GATE_FITNESS_ALPHA": "0.3",
        "DARWINX_GATE_VERIFIER_MODEL": "gpt-5.5",
    }
    assert DarwinXConfig().to_driver_env() == {}    # nothing set → driver defaults stand


def test_to_driver_env_covers_all_typed_categories() -> None:
    # One knob per emitted type across the categories (gate/scope/equivalence/heldout/verifier,
    # the DARWINX_EVAL_* runtime/cluster knobs, and trace-QC) → its driver env var, right form.
    cfg = DarwinXConfig(
        anti_cheat=True,                    # bool → "1"
        max_deletions=40,                   # int
        cross_bench_margin=0.05,            # float
        scope_mode="hard",                  # str
        heldout_benchmark="swe-bench-verified",
        clusters="a,b",                     # DARWINX_EVAL_* runtime knob (agent-agnostic name)
        absorb_timeouts=False,              # DARWINX_EVAL_* bool → "0"
        xrlenv_group_id="grp-1",            # XRLENV_-prefixed, but a per-run choice → config
        trace_qc=True,                      # DARWINX_EVOLVE_TRACE_QC
        trace_analyzer_llm_backoff_s=1.5,   # DARWINX_TRACE_* float
    )
    assert cfg.to_driver_env() == {
        "DARWINX_GATE_ANTI_CHEAT_ENABLED": "1",
        "DARWINX_GATE_MAX_DELETIONS": "40",
        "DARWINX_GATE_CROSS_BENCH_MARGIN": "0.05",
        "DARWINX_GATE_SCOPE_MODE": "hard",
        "DARWINX_GATE_HELDOUT_BENCHMARK": "swe-bench-verified",
        "DARWINX_EVAL_CLUSTERS": "a,b",
        "DARWINX_EVAL_ABSORB_TIMEOUTS": "0",
        "XRLENV_GROUP_ID": "grp-1",
        "DARWINX_EVOLVE_TRACE_QC": "1",
        "DARWINX_TRACE_LLM_BACKOFF_S": "1.5",
    }


def test_removed_dead_qd_knobs_fail_loud() -> None:
    # The old speculative QD knobs (never read by the vendored driver) were removed; setting one
    # now trips the drift guard instead of silently doing nothing.
    for dead in ("population_size", "children_per_gen", "max_generations", "patience", "qd_archive"):
        with pytest.raises(ValidationError):
            DarwinXConfig(**{dead: 8})


def test_pipelineconfig_loop_knobs_are_not_emitted_as_env() -> None:
    # Loop knobs that mirror the driver's PipelineConfig reach it BY NAME (build_pipeline_config),
    # so to_driver_env must NOT also emit them as env (that path is env-only knobs).
    assert DarwinXConfig(max_loop_iters=3, mini_eval_k_samples=5,
                         guard_enabled=True).to_driver_env() == {}


def test_evolvee_effort_translates_to_monet_eval_effort() -> None:
    # The evolvee's monet reasoning effort → the driver's DARWINX_EVAL_EFFORT env, which
    # build_codingbench_config reads → monet --effort. Unset → the driver's default (`none`).
    assert DarwinXConfig(evolvee_effort="high").to_driver_env() == {"DARWINX_EVAL_EFFORT": "high"}
    assert "DARWINX_EVAL_EFFORT" not in DarwinXConfig().to_driver_env()
    with pytest.raises(ValidationError):        # drift guard: only monet's known effort levels
        DarwinXConfig(evolvee_effort="turbo")


def test_darwinx_declares_its_config_class() -> None:
    assert bgl.algorithms.DarwinX.Config is DarwinXConfig
    assert issubclass(DarwinXConfig, AlgorithmConfig)


def test_trivial_algorithm_needs_no_config_subclass() -> None:
    class _Toy(EvolveAlgorithm):
        def evolve(self, *, evaluate, evolvee, evolver, val=None, config=None):  # noqa: ANN001
            return None

    toy = _Toy(some_knob=5)          # base Config (permissive) accepts it
    assert toy.hparams == {"some_knob": 5}
