"""Unit tests for JsonlTrajectoryReader (spec 17 §"Sink-aware reader").

Exercises all three range_kind modes, boundary edge cases, and the unknown
rollout_id error path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from xrlenv.backends.base import ResourceSpec
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateManifest,
)
from xrlenv.control.trajectory_sink import PlatformJsonlSink
from xrlenv.node.trajectory_reader import JsonlTrajectoryReader
from xrlenv.types import RolloutStatus, Step

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_N_STEPS = 4


def _manifest() -> TemplateManifest:
    return TemplateManifest(
        name="t", version="0.1", digest="sha256:t", image="im:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


def _write_sealed_run(runs_root: Path, rollout_id: str, n_steps: int = _N_STEPS) -> None:
    sink = PlatformJsonlSink(runs_root)
    sink.open(rollout_id=rollout_id, manifest=_manifest(), init={}, node_id="n")
    for i in range(n_steps):
        sink.record_step(rollout_id, Step(
            index=i, action={"a": i}, obs={"o": i}, reward=float(i),
            done=(i == n_steps - 1), truncated=False, info={}, ts=float(i),
        ))
    sink.seal(
        rollout_id=rollout_id, status=RolloutStatus.FINISHED,
        reason=None, final_reward=float(n_steps - 1), metadata={},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    return root


@pytest.fixture
def reader(runs_root: Path) -> JsonlTrajectoryReader:
    _write_sealed_run(runs_root, "r-001")
    return JsonlTrajectoryReader(runs_root)


# ──────────────────────────────────────────────────────────────────────────────
# range_kind parametrize
# ──────────────────────────────────────────────────────────────────────────────


def test_whole_returns_all_steps(reader: JsonlTrajectoryReader) -> None:
    t = reader.read_range("r-001", range_kind="whole")
    assert len(t.steps) == _N_STEPS
    assert t.steps[0].index == 0
    assert t.steps[-1].index == _N_STEPS - 1


def test_summary_only_returns_no_steps_but_preserves_metadata(
    reader: JsonlTrajectoryReader,
) -> None:
    t = reader.read_range("r-001", range_kind="summary_only")
    assert t.steps == []
    assert t.rollout_id == "r-001"
    assert t.template == "t"
    assert t.status == RolloutStatus.FINISHED


def test_step_range_slices_correctly(reader: JsonlTrajectoryReader) -> None:
    t = reader.read_range("r-001", range_kind="step_range", step_start=1, step_end=3)
    assert len(t.steps) == 2
    assert t.steps[0].index == 1
    assert t.steps[1].index == 2


# ──────────────────────────────────────────────────────────────────────────────
# step_end=None and step_end=0 mean "to end of trajectory"
# ──────────────────────────────────────────────────────────────────────────────


def test_step_range_step_end_none_means_to_end(reader: JsonlTrajectoryReader) -> None:
    t = reader.read_range("r-001", range_kind="step_range", step_start=2, step_end=None)
    assert len(t.steps) == _N_STEPS - 2
    assert t.steps[0].index == 2


def test_step_range_step_end_zero_means_to_end(reader: JsonlTrajectoryReader) -> None:
    t = reader.read_range("r-001", range_kind="step_range", step_start=1, step_end=0)
    assert len(t.steps) == _N_STEPS - 1
    assert t.steps[0].index == 1


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────────────


def test_step_range_start_beyond_total_returns_empty(
    reader: JsonlTrajectoryReader,
) -> None:
    t = reader.read_range(
        "r-001", range_kind="step_range", step_start=100, step_end=None
    )
    assert t.steps == []


def test_step_range_start_equals_end_returns_empty(
    reader: JsonlTrajectoryReader,
) -> None:
    t = reader.read_range("r-001", range_kind="step_range", step_start=2, step_end=2)
    assert t.steps == []


def test_whole_default_range_kind(reader: JsonlTrajectoryReader) -> None:
    t = reader.read_range("r-001")
    assert len(t.steps) == _N_STEPS


# ──────────────────────────────────────────────────────────────────────────────
# Unknown rollout_id
# ──────────────────────────────────────────────────────────────────────────────


def test_unknown_rollout_id_raises_file_not_found(reader: JsonlTrajectoryReader) -> None:
    with pytest.raises(FileNotFoundError):
        reader.read_range("r-does-not-exist")
