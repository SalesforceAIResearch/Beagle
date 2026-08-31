"""Unit tests for the per-user symlink workspace (no dataset copy/writes)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import workspace


def _make_dataset(root: Path, repo: str) -> None:
    d = root / repo
    (d / "srs" / "milestone_001").mkdir(parents=True)
    (d / "test_results").mkdir(parents=True)
    # dockerfiles/<mid>/test_config.json is the per-milestone test config; if it's
    # not linked, EvoClaw silently falls back to a pytest default (wrong framework).
    (d / "dockerfiles" / "milestone_001").mkdir(parents=True)
    (d / "dockerfiles" / "milestone_001" / "test_config.json").write_text("[]")
    # a stale e2e_trial from a prior dataset run — MUST NOT be linked.
    (d / "e2e_trial" / "old").mkdir(parents=True)
    for f in ("metadata.json", "dependencies.csv", "milestones.csv",
              "selected_milestone_ids.txt", "e2e_config.yaml"):
        (d / f).write_text("x")
    (d / "srs" / "milestone_001" / "SRS.md").write_text("srs")
    (root / "config").mkdir()
    (root / "config" / f"{repo}.yaml").write_text("cfg")


def test_link_workspace_symlinks_reads_no_copy(tmp_path):
    data_root = tmp_path / "EvoClaw-data"
    base = tmp_path / "myworkspaces"
    repo = "navidrome_navidrome_v0.57.0_v0.58.0"
    _make_dataset(data_root, repo)

    ws = workspace.link_workspace(data_root, base, repo)

    assert ws == base / repo
    # read entries are symlinks into the shared dataset (not copies) — including
    # dockerfiles/ (holds each milestone's test_config.json).
    for name in ("metadata.json", "dependencies.csv", "milestones.csv",
                 "selected_milestone_ids.txt", "e2e_config.yaml", "srs",
                 "test_results", "dockerfiles"):
        link = ws / name
        assert link.is_symlink()
        assert link.resolve() == (data_root / repo / name).resolve()
    # test_config.json is reachable through the dockerfiles symlink
    assert (ws / "dockerfiles" / "milestone_001" / "test_config.json").read_text() == "[]"
    # the dataset's own e2e_trial (a write target) is NOT linked
    assert not (ws / "e2e_trial").exists()
    # repo config linked at the PARENT level (evaluator reads ws.parent/config/<repo>.yaml)
    cfg = base / "config" / f"{repo}.yaml"
    assert cfg.is_symlink() and cfg.resolve() == (data_root / "config" / f"{repo}.yaml").resolve()
    # data read through the symlink works
    assert (ws / "srs" / "milestone_001" / "SRS.md").read_text() == "srs"


def test_link_workspace_writes_stay_local(tmp_path):
    data_root = tmp_path / "EvoClaw-data"
    base = tmp_path / "ws"
    repo = "r"
    _make_dataset(data_root, repo)
    ws = workspace.link_workspace(data_root, base, repo)
    # e2e_trial is a real local dir, NOT a symlink into the shared dataset's own
    # e2e_trial (which _make_dataset seeded with `old/`).
    assert not (ws / "e2e_trial").is_symlink()
    (ws / "e2e_trial" / "new").mkdir(parents=True)
    assert (base / repo / "e2e_trial" / "new").is_dir()
    # the shared dataset's e2e_trial still has only its pre-existing content
    assert (data_root / repo / "e2e_trial" / "old").is_dir()
    assert not (data_root / repo / "e2e_trial" / "new").exists()  # dataset untouched


def test_link_workspace_idempotent(tmp_path):
    data_root = tmp_path / "d"
    base = tmp_path / "b"
    _make_dataset(data_root, "r")
    workspace.link_workspace(data_root, base, "r")
    workspace.link_workspace(data_root, base, "r")  # re-run: no raise
    assert (base / "r" / "metadata.json").is_symlink()


def test_link_workspace_refuses_to_clobber_real_file(tmp_path):
    data_root = tmp_path / "d"
    base = tmp_path / "b"
    _make_dataset(data_root, "r")
    (base / "r").mkdir(parents=True)
    (base / "r" / "metadata.json").write_text("REAL")  # a real file, not a symlink
    with pytest.raises(SystemExit, match="not a symlink"):
        workspace.link_workspace(data_root, base, "r")


def test_link_workspace_missing_repo_data_fails(tmp_path):
    with pytest.raises(SystemExit, match="no repo data"):
        workspace.link_workspace(tmp_path / "d", tmp_path / "b", "nope")


def test_single_milestone_workspace_writes_one_id(tmp_path):
    data_root = tmp_path / "EvoClaw-data"
    base = tmp_path / "pm"
    repo = "navidrome_navidrome_v0.57.0_v0.58.0"
    _make_dataset(data_root, repo)
    # dataset's selected set has several ids; the per-milestone ws must override it
    (data_root / repo / "selected_milestone_ids.txt").write_text(
        "milestone_001\nmilestone_002\nmilestone_003\n"
    )
    ws = workspace.link_workspace(data_root, base, repo, single_milestone="milestone_002")
    sel = ws / "selected_milestone_ids.txt"
    # a REAL one-line file (not a symlink into the shared dataset)
    assert not sel.is_symlink()
    assert sel.read_text().split() == ["milestone_002"]
    # everything else still symlinked (e.g. dockerfiles, metadata.json)
    assert (ws / "metadata.json").is_symlink()
    assert (ws / "dockerfiles").is_symlink()
    # ws.name is the real repo (golden extraction derives the image from it)
    assert ws.name == repo
