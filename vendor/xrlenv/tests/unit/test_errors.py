"""Tests for the SDK exception hierarchy (spec 05)."""

from __future__ import annotations

from xrlenv.errors import (
    BackendCapabilityMissing,
    CapacityExhausted,
    ControlPlaneLost,
    ManifestInvalid,
    NodeLost,
    RewardFnRequired,
    RolloutCancelled,
    RolloutFailed,
    RolloutTruncated,
    TemplateUnknown,
    XRLEnvError,
)


def test_user_errors_have_user_category() -> None:
    for cls in (
        TemplateUnknown,
        RewardFnRequired,
        BackendCapabilityMissing,
        ManifestInvalid,
    ):
        assert cls.category == "user"
        assert cls.retryable is False


def test_infra_errors_are_retryable() -> None:
    for cls in (
        CapacityExhausted,
        ControlPlaneLost,
        NodeLost,
    ):
        assert cls.category == "infra"
        assert cls.retryable is True


def test_workload_errors_carry_partial_trajectory() -> None:
    err = RolloutTruncated("hard deadline hit", partial=None)
    assert err.category == "workload"
    assert err.partial is None

    cancelled = RolloutCancelled("consumer cancelled")
    assert cancelled.category == "workload"

    failed = RolloutFailed("setup blew up", reason="setup_failed")
    assert failed.reason == "setup_failed"


def test_xrlenvchain_inheritance() -> None:
    assert issubclass(TemplateUnknown, XRLEnvError)
    assert issubclass(RolloutFailed, XRLEnvError)
