"""Tests for the mixture knobs reaching the driver, and for validating the campaign winner.

These are the two halves of "generally better, not overfit": the driver gates candidates
in-loop from DARWINX_GATE_MIXTURE_*, and beagle checks the winner on held-out data afterwards.
"""

from __future__ import annotations

from beagle.algorithms.base import Candidate, CandidateStatus
from beagle.agents.core.spec import AgentSource
from beagle.algorithms.darwinx.config import DarwinXConfig
from beagle.types import TaskResult


def _cfg(**kw) -> DarwinXConfig:
    return DarwinXConfig(repo_root="/tmp/repo", **kw)


# -- knobs reaching the driver ---------------------------------------------------------


def test_mixture_knobs_are_off_unless_set():
    # The single-benchmark arms must be byte-for-byte unchanged, so an unset knob emits
    # nothing and the driver's own default stands.
    env = _cfg().to_driver_env()
    assert not [k for k in env if k.startswith("DARWINX_GATE_MIXTURE_")]


def test_mixture_knobs_map_to_the_drivers_env():
    env = _cfg(
        mixture_gate=True,
        mixture_spec="@/tmp/monet_mixture_spec.json",
        mixture_tol_sd=1.0,
        mixture_min_abs_drop=0.02,
        mixture_gain_cap_sd=2.0,
    ).to_driver_env()
    assert env["DARWINX_GATE_MIXTURE_GATE"] == "1"
    assert env["DARWINX_GATE_MIXTURE_SPEC"].endswith("monet_mixture_spec.json")
    assert env["DARWINX_GATE_MIXTURE_TOL_SD"] == "1.0"
    assert env["DARWINX_GATE_MIXTURE_MIN_ABS_DROP"] == "0.02"
    assert env["DARWINX_GATE_MIXTURE_GAIN_CAP_SD"] == "2.0"


def test_mixture_gate_can_be_explicitly_disabled():
    assert _cfg(mixture_gate=False).to_driver_env()["DARWINX_GATE_MIXTURE_GATE"] == "0"


def test_an_unknown_mixture_knob_is_a_load_time_error():
    # DarwinXConfig forbids extras: a typo'd knob must fail loudly rather than sit inert for
    # a whole campaign.
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _cfg(mixture_gain_cap=2.0)   # real field is mixture_gain_cap_sd


# -- validating the winner -------------------------------------------------------------


class _Algo:
    """Just the _validate half of DarwinX, without launching a campaign."""

    from beagle.algorithms.darwinx.algorithm import DarwinX as _DarwinX
    _validate = _DarwinX._validate


def _candidate() -> Candidate:
    return Candidate(id="c1", source=AgentSource(repo="r", ref="main"),
                     status=CandidateStatus.EVALUATED, score=0.61,
                     results=[TaskResult(task_id="train1", benchmark="b", resolved=True)])


def test_validation_records_per_benchmark_signals_on_the_winner():
    cand = _candidate()

    def val(c):
        c.score = 0.44
        c.results = [
            TaskResult(task_id="v1", benchmark="swe-bench-verified", resolved=True),
            TaskResult(task_id="v2", benchmark="swe-bench-verified", resolved=False),
            TaskResult(task_id="d1", benchmark="deep-swe", resolved=True),
        ]

    _Algo()._validate(cand, val)
    v = cand.metadata["validation"]
    assert v["values"] == {"swe-bench-verified": 0.5, "deep-swe": 1.0}


def test_validation_does_not_overwrite_the_campaign_score():
    # val and evaluate are the same closure shape and both write score/results in place.
    cand = _candidate()

    def val(c):
        c.score = 0.44
        c.results = [TaskResult(task_id="v1", benchmark="b", resolved=True)]

    _Algo()._validate(cand, val)
    assert cand.score == 0.61
    assert [r.task_id for r in cand.results] == ["train1"]


def test_a_broken_validator_does_not_lose_the_campaign():
    # Hours of compute produced this candidate; a validator that raises must not discard it.
    cand = _candidate()

    def val(c):
        raise RuntimeError("cluster went away")

    _Algo()._validate(cand, val)
    assert "validation_error" in cand.metadata
    assert cand.score == 0.61


def test_infra_errors_in_validation_are_not_counted_as_losses():
    cand = _candidate()

    def val(c):
        c.results = [
            TaskResult(task_id="v1", benchmark="swe-bench-verified", resolved=True),
            TaskResult(task_id="v2", benchmark="swe-bench-verified", error="HTTP 502"),
        ]

    _Algo()._validate(cand, val)
    # 1 of 1 measured, not 1 of 2 -- the outage is not the harness's fault.
    assert cand.metadata["validation"]["values"]["swe-bench-verified"] == 1.0
