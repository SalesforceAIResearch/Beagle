"""Tests for the platform-jsonl trajectory sink (spec 08, spec 20).

Covers the on-disk run-dir layout, per-step append, seal-time meta update,
read-back, and the cooperation with the coordinator's replay path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from xrlenv.backends.base import ResourceSpec
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateManifest,
)
from xrlenv.control.trajectory_sink import (
    PlatformJsonlSink,
    TrajectoryLocator,
    run_dir_for,
)
from xrlenv.types import RolloutStatus, Step


def _manifest(name: str = "hello-shell") -> TemplateManifest:
    return TemplateManifest(
        name=name,
        version="0.1",
        digest="sha256:deadbeef",
        image="im:1",
        resources=ResourceSpec(
            cpu_request=0.25,
            cpu_limit=1.0,
            mem_request_bytes=64_000_000,
            mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


def _step(index: int = 0, reward: float = 0.0) -> Step:
    return Step(
        index=index,
        action={"cmd": f"echo step-{index}"},
        obs={"stdout": f"step-{index}\n", "exit_code": 0},
        reward=reward,
        done=False,
        truncated=False,
        info={"steps": index + 1},
        ts=float(index) * 0.1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Run-dir layout
# ──────────────────────────────────────────────────────────────────────────────


def test_run_dir_for_uses_utc_date(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    when = datetime(2026, 4, 26, 12, 0, 0, tzinfo=UTC)
    p = run_dir_for(tmp_path / "runs", "r-1", when=when)
    assert p == tmp_path / "runs" / "2026-04-26" / "r-1"


# ──────────────────────────────────────────────────────────────────────────────
# Sink lifecycle
# ──────────────────────────────────────────────────────────────────────────────


def test_open_creates_dir_and_writes_initial_meta(tmp_path: Path) -> None:
    sink = PlatformJsonlSink(tmp_path / "runs")
    locator = sink.open(
        rollout_id="r-1",
        manifest=_manifest(),
        init={"max_steps": 3},
        node_id="local",
    )

    assert isinstance(locator, TrajectoryLocator)
    assert locator.sink == "platform-jsonl"
    assert locator.node_id == "local"
    assert locator.uri is not None
    assert locator.uri.startswith("file://")

    # meta.json exists with the running-state envelope
    meta_path = next(tmp_path.glob("runs/*/r-1/meta.json"))
    meta = json.loads(meta_path.read_text())
    assert meta["rollout_id"] == "r-1"
    assert meta["template"] == "hello-shell"
    assert meta["template_digest"] == "sha256:deadbeef"
    assert meta["status"] == RolloutStatus.RUNNING.value
    assert meta["init"] == {"max_steps": 3}


def test_run_dir_for_rollout_returns_open_path(tmp_path: Path) -> None:
    """``run_dir_for_rollout`` returns the live run dir for an open
    rollout — coordinator's reward path uses this to drop a
    ``verifier/`` directory mirroring harbor's trial layout."""
    sink = PlatformJsonlSink(tmp_path / "runs")
    sink.open(
        rollout_id="r-vd1", manifest=_manifest(), init={}, node_id="local",
    )
    rd = sink.run_dir_for_rollout("r-vd1")
    assert rd is not None
    assert rd.is_dir()
    assert rd.name == "r-vd1"
    # Sibling files (jsonl + meta) live under the same dir.
    assert (rd / "trajectory.jsonl").parent == rd


def test_run_dir_for_rollout_returns_sealed_path(tmp_path: Path) -> None:
    """After seal the rollout drops out of the in-memory map; the
    lookup falls back to walking ``runs_root`` so the reward path
    still finds the dir during late-running operations."""
    sink = PlatformJsonlSink(tmp_path / "runs")
    sink.open(
        rollout_id="r-vd2", manifest=_manifest(), init={}, node_id="local",
    )
    sink.seal(
        rollout_id="r-vd2",
        status=RolloutStatus.FINISHED,
        reason=None,
        final_reward=1.0,
        metadata={},
    )
    rd = sink.run_dir_for_rollout("r-vd2")
    assert rd is not None
    assert rd.name == "r-vd2"


def test_run_dir_for_rollout_returns_none_for_unknown(tmp_path: Path) -> None:
    sink = PlatformJsonlSink(tmp_path / "runs")
    assert sink.run_dir_for_rollout("never-existed") is None


def test_record_step_appends_jsonl(tmp_path: Path) -> None:
    sink = PlatformJsonlSink(tmp_path / "runs")
    sink.open(rollout_id="r-2", manifest=_manifest(), init={}, node_id="local")
    sink.record_step("r-2", _step(0, reward=0.5))
    sink.record_step("r-2", _step(1, reward=0.25))

    jsonl_path = next(tmp_path.glob("runs/*/r-2/trajectory.jsonl"))
    lines = jsonl_path.read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["index"] == 0
    assert parsed[0]["reward"] == 0.5
    assert parsed[1]["index"] == 1


def test_seal_updates_meta_and_returns_final_locator(tmp_path: Path) -> None:
    sink = PlatformJsonlSink(tmp_path / "runs")
    sink.open(rollout_id="r-3", manifest=_manifest(), init={}, node_id="local")
    sink.record_step("r-3", _step(0, reward=1.0))
    locator = sink.seal(
        rollout_id="r-3",
        status=RolloutStatus.FINISHED,
        reason=None,
        final_reward=1.0,
        metadata={"backend": "docker"},
    )

    assert locator.size_bytes is not None
    assert locator.size_bytes > 0

    meta = json.loads(next(tmp_path.glob("runs/*/r-3/meta.json")).read_text())
    assert meta["status"] == RolloutStatus.FINISHED.value
    assert meta["final_reward"] == 1.0
    assert meta["step_count"] == 1
    assert meta["metadata"] == {"backend": "docker"}
    assert meta["ended_at"] is not None


def test_record_step_before_open_raises(tmp_path: Path) -> None:
    sink = PlatformJsonlSink(tmp_path / "runs")
    with pytest.raises(KeyError):
        sink.record_step("not-open", _step(0))


# ──────────────────────────────────────────────────────────────────────────────
# Read-back
# ──────────────────────────────────────────────────────────────────────────────


def test_read_returns_full_trajectory(tmp_path: Path) -> None:
    sink = PlatformJsonlSink(tmp_path / "runs")
    sink.open(rollout_id="r-4", manifest=_manifest(), init={}, node_id="local")
    sink.record_step("r-4", _step(0, reward=0.4))
    sink.record_step("r-4", _step(1, reward=0.6))
    sink.seal(
        rollout_id="r-4",
        status=RolloutStatus.FINISHED,
        reason=None,
        final_reward=1.0,
        metadata={"k": "v"},
    )

    traj = sink.read("r-4")
    assert traj.rollout_id == "r-4"
    assert traj.template == "hello-shell"
    assert traj.status == RolloutStatus.FINISHED
    assert traj.final_reward == 1.0
    assert len(traj.steps) == 2
    assert traj.steps[0].action == {"cmd": "echo step-0"}
    # Metadata carries the user-supplied dict plus the rollout's home
    # node_id (folded in from the top-level meta.json field so callers
    # don't have to dig into the seal-time state record).
    assert traj.metadata == {"k": "v", "node_id": "local"}


def test_read_unknown_rollout_raises(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    sink = PlatformJsonlSink(tmp_path / "runs")
    with pytest.raises(FileNotFoundError):
        sink.read("never-existed")


def test_read_in_flight_returns_running_status(tmp_path: Path) -> None:
    sink = PlatformJsonlSink(tmp_path / "runs")
    sink.open(rollout_id="r-5", manifest=_manifest(), init={}, node_id="local")
    sink.record_step("r-5", _step(0, reward=0.0))
    # No seal yet — read should still work and report RUNNING.
    traj = sink.read("r-5")
    assert traj.status == RolloutStatus.RUNNING
    assert len(traj.steps) == 1


def test_open_truncates_prior_body(tmp_path: Path) -> None:
    sink = PlatformJsonlSink(tmp_path / "runs")
    # First open writes one step.
    sink.open(rollout_id="r-6", manifest=_manifest(), init={}, node_id="local")
    sink.record_step("r-6", _step(0, reward=0.0))
    sink.seal(
        rollout_id="r-6",
        status=RolloutStatus.FINISHED,
        reason=None,
        final_reward=0.0,
        metadata={},
    )
    # Re-open with same id (operator-driven re-run): body must be empty again.
    sink.open(rollout_id="r-6", manifest=_manifest(), init={}, node_id="local")
    jsonl = next(tmp_path.glob("runs/*/r-6/trajectory.jsonl"))
    assert jsonl.read_bytes() == b""
