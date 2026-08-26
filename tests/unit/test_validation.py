"""Tests for the validation alignment layer.

The properties pinned here are the ones that make a heterogeneous mixture safe to select on:
an unmeasured trial never becomes a loss, lower-is-better metrics land in the same signed
space as pass rate, and an uncalibrated slice is excluded rather than given a default.
"""

from __future__ import annotations

import pytest

from beagle.eval.validation import (
    Direction,
    Outcome,
    OutcomePolicy,
    Reference,
    ValReport,
    align,
    capture,
    classify,
    mean_tokens,
    mean_turns,
    pass_rate,
    summarize,
)
from beagle.types import TaskResult


def R(task_id, benchmark="b", *, resolved=False, reward=None, error=None,
      turns=0, duration=0.0, tokens=None) -> TaskResult:
    return TaskResult(task_id=task_id, benchmark=benchmark, resolved=resolved, reward=reward,
                      error=error, num_turns=turns, duration_sec=duration, tokens=tokens or {})


# -- classification --------------------------------------------------------------------


def test_clean_pass_and_clean_fail():
    assert classify(R("t", resolved=True)) is Outcome.SOLVED
    assert classify(R("t", resolved=False)) is Outcome.FAILED


def test_infra_error_is_unmeasured_not_failed():
    # The whole point: our outages must not become evolutionary pressure.
    for err in ("ConnectionError: gateway down", "HTTP 401", "DockerException: ...",
                "PermissionError(13)"):
        assert classify(R("t", error=err)) is Outcome.UNMEASURED


def test_timeout_is_a_measurement_not_an_outage():
    # The agent ran and blew its budget; that is signal, not missing data.
    assert classify(R("t", error="AgentTimeoutError")) is Outcome.FAILED


def test_timeout_policy_is_explicit():
    lenient = OutcomePolicy(timeout_is_failure=False)
    # With timeouts not counted as failures they fall through to the grader's verdict.
    assert classify(R("t", resolved=True, error="AgentTimeoutError"), lenient) is Outcome.SOLVED


def test_verifier_passed_but_errored_is_not_a_win_by_default():
    assert classify(R("t", reward=1.0, error="AgentTimeoutError")) is Outcome.FAILED


def test_unknown_error_is_unmeasured_rather_than_a_loss():
    # Guessing wrong here costs a data point; guessing the other way feeds breakage into
    # selection, which is strictly worse.
    assert classify(R("t", error="something nobody has seen before")) is Outcome.UNMEASURED


def test_in_band_reward_threshold():
    assert classify(R("t", reward=1.0)) is Outcome.SOLVED
    assert classify(R("t", reward=0.5)) is Outcome.FAILED


# -- pass rate -------------------------------------------------------------------------


def _graded(results, policy=None):
    return [(r, classify(r, policy)) for r in results]


def test_pass_rate_excludes_unmeasured_from_the_denominator():
    graded = _graded([
        R("t1", resolved=True),
        R("t2", resolved=False),
        R("t3", error="HTTP 502"),          # unmeasured
    ])
    m = pass_rate(graded)
    assert m is not None
    assert m.value == pytest.approx(0.5)     # 1 of 2 measured, not 1 of 3
    assert (m.n, m.unmeasured) == (2, 1)


def test_pass_rate_averages_per_task_then_across_tasks():
    # t1 ran 4 times (retries), t2 once. Averaging trials would let t1 dominate.
    graded = _graded([
        R("t1", resolved=True), R("t1", resolved=True),
        R("t1", resolved=True), R("t1", resolved=False),
        R("t2", resolved=False),
    ])
    m = pass_rate(graded)
    assert m is not None
    assert m.value == pytest.approx((0.75 + 0.0) / 2)
    assert (m.n, m.k) == (2, 4)


def test_pass_rate_is_none_when_nothing_was_measured():
    assert pass_rate(_graded([R("t1", error="gateway"), R("t2", error="tunnel")])) is None


def test_coverage_reports_how_much_was_lost():
    graded = _graded([R("t1", resolved=True), R("t2", error="HTTP 503")])
    m = pass_rate(graded)
    assert m is not None and m.coverage == pytest.approx(0.5)


# -- efficiency metrics ----------------------------------------------------------------


def test_efficiency_metrics_are_lower_is_better():
    graded = _graded([R("t1", resolved=True, turns=10), R("t2", resolved=True, turns=20)])
    m = mean_turns(graded)
    assert m is not None
    assert m.value == pytest.approx(15.0)
    assert m.direction is Direction.LOWER_IS_BETTER


def test_tokens_sum_across_the_token_counts():
    graded = _graded([R("t1", resolved=True, tokens={"input": 100, "output": 50})])
    m = mean_tokens(graded)
    assert m is not None and m.value == pytest.approx(150.0)


def test_efficiency_ignores_unmeasured_trials():
    graded = _graded([R("t1", resolved=True, turns=10), R("t2", error="HTTP 500", turns=999)])
    m = mean_turns(graded)
    assert m is not None and m.value == pytest.approx(10.0)


# -- alignment -------------------------------------------------------------------------


def test_higher_is_better_aligns_positively_when_it_rises():
    graded = _graded([R("t1", resolved=True)])
    m = pass_rate(graded)
    assert m is not None
    sig = align("b", m, Reference(baseline=0.5, sd=0.1))
    assert sig.sigma == pytest.approx(5.0)


def test_lower_is_better_aligns_positively_when_it_falls():
    # A harness that got cheaper must read as an improvement, in the same units as accuracy.
    graded = _graded([R("t1", resolved=True, turns=8)])
    m = mean_turns(graded)
    assert m is not None
    sig = align("b", m, Reference(baseline=10.0, sd=1.0))
    assert sig.sigma == pytest.approx(2.0)
    sig_worse = align("b", m, Reference(baseline=6.0, sd=1.0))
    assert sig_worse.sigma == pytest.approx(-2.0)


def test_uncalibrated_slice_has_no_sigma_and_is_flagged():
    graded = _graded([R("t1", resolved=True)])
    m = pass_rate(graded)
    assert m is not None
    sig = align("b", m, None)
    assert sig.sigma is None and not sig.calibrated


def test_sd_floor_bounds_a_zero_noise_reference():
    graded = _graded([R("t1", resolved=True)])
    m = pass_rate(graded)
    assert m is not None
    assert align("b", m, Reference(baseline=0.99, sd=0.0)).sigma == pytest.approx(1.0)


# -- the report ------------------------------------------------------------------------


def _mixture_results():
    return [
        R("v1", "swe-bench-verified", resolved=True, turns=10),
        R("v2", "swe-bench-verified", resolved=False, turns=12),
        R("d1", "deep-swe", resolved=True, turns=20),
        R("d2", "deep-swe", error="HTTP 502"),
        R("p1", "swe-bench-pro", error="ConnectionError"),
    ]


def test_summarize_slices_by_benchmark_with_no_extra_bookkeeping():
    rep = summarize(_mixture_results())
    assert rep.slices == ["deep-swe", "swe-bench-verified"]
    # Pro produced nothing measurable at all -- reported, not scored as zero.
    assert rep.empty_slices == ["swe-bench-pro"]


def test_summarize_aligns_only_calibrated_slices():
    rep = summarize(
        _mixture_results(),
        references={"swe-bench-verified": Reference(0.4, 0.05)},
    )
    assert rep.sigmas() == {"swe-bench-verified": pytest.approx(2.0)}
    assert rep.uncalibrated() == ["deep-swe"]


def test_references_can_be_keyed_by_slice_and_metric():
    rep = summarize(
        _mixture_results(),
        references={("swe-bench-verified", "mean_turns"): Reference(12.0, 1.0)},
    )
    turns = rep.get("swe-bench-verified", "mean_turns")
    assert turns is not None and turns.sigma == pytest.approx(1.0)


def test_values_feed_the_aggregate_and_the_floor():
    rep = summarize(_mixture_results())
    assert rep.values() == {"swe-bench-verified": pytest.approx(0.5),
                            "deep-swe": pytest.approx(1.0)}


def test_low_coverage_slices_are_called_out():
    rep = summarize([
        R("d1", "deep-swe", resolved=True),
        R("d2", "deep-swe", error="HTTP 502"),
        R("d3", "deep-swe", error="HTTP 502"),
    ])
    assert rep.low_coverage() == ["deep-swe"]


def test_custom_slicing_needs_no_upstream_change():
    rep = summarize(_mixture_results(), slice_of=lambda r: "all")
    assert rep.slices == ["all"]


def test_feedback_names_where_it_lost():
    rep = summarize(
        _mixture_results(),
        references={"swe-bench-verified": Reference(0.9, 0.05)},
    )
    text = rep.to_feedback()
    assert "swe-bench-verified" in text and "worse" in text
    assert "uncalibrated" in text          # deep-swe
    assert "no measurement at all" in text  # swe-bench-pro


def test_empty_report_says_so():
    assert "no validation signals" in ValReport().to_feedback()


# -- capture ---------------------------------------------------------------------------


class _Candidate:
    def __init__(self):
        self.score = None
        self.results = []


def test_capture_returns_val_results_without_clobbering_the_train_score():
    # Trainer builds val with the same _make_evaluate as the training scorer, so both write
    # candidate.score/results in place; without this the held-out numbers silently replace
    # the fitness that was just measured.
    cand = _Candidate()
    cand.score, cand.results = 0.62, [R("train1", resolved=True)]

    def val(c):
        c.score = 0.31
        c.results = [R("val1", "deep-swe", resolved=True)]

    got = capture(val, cand)
    assert [r.task_id for r in got] == ["val1"]
    assert cand.score == 0.62
    assert [r.task_id for r in cand.results] == ["train1"]


def test_capture_restores_even_when_the_evaluator_raises():
    cand = _Candidate()
    cand.score, cand.results = 0.62, [R("train1", resolved=True)]

    def boom(c):
        c.score = 0.0
        raise RuntimeError("cluster went away")

    with pytest.raises(RuntimeError):
        capture(boom, cand)
    assert cand.score == 0.62
