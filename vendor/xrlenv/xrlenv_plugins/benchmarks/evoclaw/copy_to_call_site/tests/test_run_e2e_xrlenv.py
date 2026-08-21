"""Unit tests for run_e2e_xrlenv arg handling (no harness/cluster needed)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_e2e_xrlenv as entry


def _ref(_repo_name):
    return "hyd2apse/navidrome:base-v0.9"


# --- --workspace-root must hold the repo data (arbitrary dir rejected) ---
def test_validate_workspace_root_arbitrary_dir_fails_loud(tmp_path):
    with pytest.raises(SystemExit, match="missing EvoClaw repo data"):
        entry._validate_workspace_root(tmp_path)  # empty -> rejected


def test_validate_workspace_root_with_data_ok(tmp_path):
    for f in ("metadata.json", "dependencies.csv", "milestones.csv"):
        (tmp_path / f).write_text("{}")
    entry._validate_workspace_root(tmp_path)  # no raise


# --- derivation: srs from explicit workspace-root, image from repo-name ---
def test_derives_srs_from_workspace_and_image_from_repo():
    argv = ["--repo-name", "r", "--workspace-root", "/ws", "--agent", "oracle"]
    assert entry._derive_default_args(argv, _ref) == [
        "--srs-root", "/ws/srs",
        "--image", "hyd2apse/navidrome:base-v0.9",
    ]


def test_workspace_root_never_derived():
    # No --workspace-root -> srs is NOT derived (workspace-root is not invented);
    # only --image is filled from --repo-name.
    assert entry._derive_default_args(["--repo-name", "r"], _ref) == [
        "--image", "hyd2apse/navidrome:base-v0.9",
    ]


def test_respects_explicit_srs_and_image():
    argv = ["--repo-name", "r", "--workspace-root", "/ws", "--srs-root", "/s", "--image", "img:1"]
    assert entry._derive_default_args(argv, _ref) == []


def test_unknown_repo_skips_image():
    argv = ["--repo-name", "r", "--workspace-root", "/ws"]
    assert entry._derive_default_args(argv, lambda n: None) == ["--srs-root", "/ws/srs"]


# --- fast eval config zeroes the debounce/recovery waits ---
def test_fast_config_dict_zeroes_timing():
    base = {"retry_and_timing": {"debounce_seconds": 120, "max_debounce_wait": 360,
                                 "recovery_wait_seconds": 60, "evaluation_timeout": 3600},
            "early_unblock": True}
    out = entry._fast_config_dict(base)
    assert out["retry_and_timing"]["debounce_seconds"] == 0
    assert out["retry_and_timing"]["max_debounce_wait"] == 0
    assert out["retry_and_timing"]["recovery_wait_seconds"] == 0
    assert out["retry_and_timing"]["evaluation_timeout"] == 3600   # other keys kept
    assert out["early_unblock"] is True                            # rest of config kept
    assert base["retry_and_timing"]["debounce_seconds"] == 120     # input not mutated


def test_fast_config_dict_empty_base():
    assert entry._fast_config_dict({})["retry_and_timing"] == entry._FAST_TIMING


# --- agent container is torn down (else the cluster watchdog reaps the orphan) ---
# Opt-out is the --keep-container flag (via _CFG.keep_container), not an env var.
def test_inject_remove_container_appends(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--agent", "oracle"])
    monkeypatch.setattr(entry, "_CFG", argparse.Namespace(keep_container=False))
    entry._inject_remove_container()
    assert "--remove-container" in sys.argv


def test_inject_remove_container_idempotent(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--remove-container"])
    monkeypatch.setattr(entry, "_CFG", argparse.Namespace(keep_container=False))
    entry._inject_remove_container()
    assert sys.argv.count("--remove-container") == 1  # not duplicated


def test_inject_remove_container_opt_out(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--agent", "oracle"])
    monkeypatch.setattr(entry, "_CFG", argparse.Namespace(keep_container=True))
    entry._inject_remove_container()
    assert "--remove-container" not in sys.argv  # kept for inspection


# --- fleet footprint resolution (opt-in, both-or-neither) ---
def test_resolve_fleet_footprint_off_when_neither():
    assert entry._resolve_fleet_footprint(None, None, "xrl-1-") == (None, None, None)


def test_resolve_fleet_footprint_on_when_both():
    fid, cpu, mem = entry._resolve_fleet_footprint(18.0, 40.0, "xrl-1-")
    assert fid == "xrl-1-"
    assert cpu == 18.0
    assert mem == int(40.0 * 1024**3)


@pytest.mark.parametrize("cpu,mem_gb", [(18.0, None), (None, 40.0)])
def test_resolve_fleet_footprint_partial_fails_loud(cpu, mem_gb):
    with pytest.raises(SystemExit, match="TOGETHER"):
        entry._resolve_fleet_footprint(cpu, mem_gb, "xrl-1-")


@pytest.mark.parametrize("cpu,mem_gb", [(0.0, 40.0), (18.0, 0.0), (-1.0, 40.0)])
def test_resolve_fleet_footprint_nonpositive_fails_loud(cpu, mem_gb):
    with pytest.raises(SystemExit, match="positive"):
        entry._resolve_fleet_footprint(cpu, mem_gb, "xrl-1-")


# --- --cpu-pinning-milestone routes ONLY matching milestones' workers ---
def _mk_task(repo, mid, pin_set):
    from pathlib import Path

    import run_all_xrlenv as ra
    return ra._Task(
        repo, mid, data_root=Path("/d"), run_ws=Path("/w"), log_dir=Path("/l"),
        fleet_cpu=None, fleet_mem_gb=None, mem_per_cpu_gb=2.0, copy_testbed=False,
        passthru=[], cpu_pinning_milestones=frozenset(pin_set),
    )


def test_cpu_pinning_milestone_matches_and_scopes():
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    # matched by bare mid, repo/mid, repo__mid; unmatched stays off
    assert _mk_task("go-zero", "M003", {"M003"}).cpu_pinning is True
    assert _mk_task("go-zero", "M003", {"go-zero/M003"}).cpu_pinning is True
    assert _mk_task("go-zero", "M003", {"go-zero__M003"}).cpu_pinning is True
    assert _mk_task("go-zero", "M004", {"M003"}).cpu_pinning is False
    # per-repo task (mid None) is never scoped
    assert _mk_task("go-zero", None, {"M003"}).cpu_pinning is False


def test_cpu_pinning_appends_worker_flag_only_when_matched():
    from pathlib import Path as _P
    on = _mk_task("go-zero", "M003", {"M003"})._worker_cmd(_P("/ws"))
    off = _mk_task("go-zero", "M004", {"M003"})._worker_cmd(_P("/ws"))
    assert "--cpu-pinning" in on
    assert "--cpu-pinning" not in off


# --- per-task .mount teardown (stops .mount/ from accumulating one dir per task) ---
def test_cleanup_oracle_mount_removes_dir(tmp_path):
    d = tmp_path / "repo__v0.9__12345"
    d.mkdir()
    (d / "m.tar").write_bytes(b"x")
    entry._ORACLE_MOUNT_DIR = d
    entry._cleanup_oracle_mount(keep=False)
    assert not d.exists()               # dir removed
    assert entry._ORACLE_MOUNT_DIR is None  # record cleared


def test_cleanup_oracle_mount_keep_preserves_dir(tmp_path):
    d = tmp_path / "repo__v0.9__12345"
    d.mkdir()
    entry._ORACLE_MOUNT_DIR = d
    entry._cleanup_oracle_mount(keep=True)  # --keep-container: leave the bind source
    assert d.exists()
    entry._ORACLE_MOUNT_DIR = None           # tidy the global for other tests


def test_cleanup_oracle_mount_noop_when_unset():
    entry._ORACLE_MOUNT_DIR = None
    entry._cleanup_oracle_mount(keep=False)  # must not raise on a non-oracle run
