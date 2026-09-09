"""A stage cap that does not reach the driver is a lost iteration.

Measured 2026-08-26: the fast loop's evolver carried ``timeout: 1800`` while its stage caps sat
higher, and that agent-level ceiling silently ate two of its three iterations -- implement, then
analyze -- each reported only as "timeout after 1800s". These tests pin the half we can pin: that
the stage caps are typed config which reaches the driver under the names it reads, and that a
non-positive value is refused rather than silently ignored.
"""

import pytest

from beagle.algorithms.darwinx.config import DarwinXConfig


def test_stage_caps_reach_the_driver_under_the_names_it_reads():
    env = DarwinXConfig(
        analyze_timeout_s=2400, implement_timeout_s=3600, review_timeout_s=2400
    ).to_driver_env()
    assert env["DARWINX_GATE_ANALYZE_TIMEOUT_S"] == "2400"
    assert env["DARWINX_GATE_IMPLEMENT_TIMEOUT_S"] == "3600"
    assert env["DARWINX_GATE_REVIEW_TIMEOUT_S"] == "2400"


def test_unset_stage_caps_are_not_emitted_so_the_driver_keeps_its_defaults():
    env = DarwinXConfig().to_driver_env()
    assert not [k for k in env if k.endswith("_TIMEOUT_S")]


@pytest.mark.parametrize("field", ["analyze_timeout_s", "implement_timeout_s", "review_timeout_s"])
def test_a_non_positive_stage_cap_is_refused_not_silently_dropped(field):
    # The driver ignores non-positive values and falls back to its own default, so accepting one
    # here would show a cap in the config that never applied.
    with pytest.raises(ValueError, match="must be > 0"):
        DarwinXConfig(**{field: 0})
