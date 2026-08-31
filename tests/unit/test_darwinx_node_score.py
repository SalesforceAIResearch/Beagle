"""A hosted campaign can say what a node's score is, and cannot say it uselessly.

The contract these pin is narrow but it is the one the drop-in exists for: every knob the driver
honours must be expressible in the typed config, because a knob that is not expressible does not
error -- the campaign just runs with the driver's default and nobody finds out until the results are
read. Here the default is single-benchmark selection, which is exactly what the mixture was set up
to avoid.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from beagle.algorithms.darwinx.config import DarwinXConfig


def _cfg(**kw):
    return DarwinXConfig(**kw)


# ── the knob reaches the driver ──────────────────────────────────────────────────────────────

def test_mixture_scoring_reaches_the_driver_env():
    env = _cfg(mixture_gate=True, node_score="mixture").to_driver_env()
    assert env["DARWINX_GATE_NODE_SCORE"] == "mixture"


def test_panel_scoring_is_expressible_too():
    env = _cfg(node_score="panel").to_driver_env()
    assert env["DARWINX_GATE_NODE_SCORE"] == "panel"


def test_unset_leaves_the_driver_default_alone():
    """Absent means absent: the driver's own default decides, and nothing is asserted over it."""
    env = _cfg().to_driver_env()
    assert "DARWINX_GATE_NODE_SCORE" not in env


def test_only_the_two_documented_values_are_accepted():
    with pytest.raises(ValidationError):
        _cfg(mixture_gate=True, node_score="fitness")


# ── the jointly-invalid case ─────────────────────────────────────────────────────────────────

def test_mixture_scoring_without_the_gate_is_rejected():
    """Individually valid, jointly useless: no gate means no fitness, so every node scores 0.0."""
    with pytest.raises(ValidationError) as exc:
        _cfg(node_score="mixture")
    assert "mixture_gate" in str(exc.value)


def test_the_error_says_what_goes_wrong_not_just_what_is_disallowed():
    with pytest.raises(ValidationError) as exc:
        _cfg(node_score="mixture", mixture_gate=False)
    msg = str(exc.value)
    assert "ties" in msg or "0.0" in msg


def test_panel_scoring_needs_no_gate():
    assert _cfg(node_score="panel").node_score == "panel"


# ── no collateral damage to the knobs it sits next to ────────────────────────────────────────

def test_it_composes_with_the_panel_knobs_the_campaign_actually_uses():
    env = _cfg(
        mixture_gate=True,
        node_score="mixture",
        defer_node_full_eval=True,
        fixed_eval_panel=True,
        eval_panel_size=40,
        mixture_gate_tasks=80,
        mixture_gate_screen_tasks=25,
        mixture_gate_tasks_per_benchmark={"deep-swe": 24},
        mixture_gate_screen_tasks_per_benchmark={"deep-swe": 8},
    ).to_driver_env()
    assert env["DARWINX_GATE_NODE_SCORE"] == "mixture"
    assert env["DARWINX_GATE_FIXED_EVAL_PANEL"] == "1"
    assert env["DARWINX_GATE_EVAL_PANEL_SIZE"] == "40"
    assert env["DARWINX_GATE_MIXTURE_GATE_TASKS_DEEP_SWE"] == "24"
    assert env["DARWINX_GATE_MIXTURE_GATE_SCREEN_TASKS_DEEP_SWE"] == "8"


def test_the_defer_panel_rule_still_fires_independently():
    with pytest.raises(ValidationError):
        _cfg(mixture_gate=True, node_score="mixture", defer_node_full_eval=True,
             fixed_eval_panel=False)
