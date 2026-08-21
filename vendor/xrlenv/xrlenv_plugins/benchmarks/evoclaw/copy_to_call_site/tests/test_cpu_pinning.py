"""Unit tests for cpu_pinning — the ``--cpu-pinning`` onboarding patch.

Verifies the runtime patch flips ``RuntimeLimits.cpu_pinning=True`` (empty and
non-empty host_config), preserves the other limits, and is idempotent — all
without touching any ``xrlenv/`` source file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("xrlenv.compat.docker_client")
import cpu_pinning


@pytest.fixture
def resolver():
    """Save/restore the compat resolver + reset the apply guard."""
    import xrlenv.compat.docker_client as dc

    orig = dc._resolve_runtime_limits
    cpu_pinning._APPLIED = False
    try:
        yield dc
    finally:
        dc._resolve_runtime_limits = orig
        cpu_pinning._APPLIED = False


def test_flips_pinning_on_when_no_other_limits(resolver):
    dc = resolver
    assert dc._resolve_runtime_limits({}) is None  # baseline: empty -> None
    cpu_pinning.apply_cpu_pinning()
    rl = dc._resolve_runtime_limits({})
    assert rl is not None and rl.cpu_pinning is True


def test_preserves_other_limits(resolver):
    dc = resolver
    hc = {"PidsLimit": 100, "ShmSize": 2048}
    base = dc._resolve_runtime_limits(hc)
    assert base is not None and base.pids_limit == 100 and base.cpu_pinning is False
    cpu_pinning.apply_cpu_pinning()
    rl = dc._resolve_runtime_limits(hc)
    assert rl.cpu_pinning is True
    assert rl.pids_limit == 100 and rl.shm_size_bytes == 2048  # untouched


def test_idempotent(resolver):
    dc = resolver
    cpu_pinning.apply_cpu_pinning()
    once = dc._resolve_runtime_limits
    cpu_pinning.apply_cpu_pinning()  # second call must not re-wrap
    assert dc._resolve_runtime_limits is once
