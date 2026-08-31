"""Unit tests for golden extraction + caching (no harness/cluster needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# oracle.py subclasses a harness agent base at import time; skip cleanly where the EvoClaw
# harness isn't installed (e.g. this file mirrored into the xrlenv_plugins reference copy).
pytest.importorskip("harness.e2e.agents.base")
import oracle


class _FakeExec:
    exit_code = 0
    output = (b"GOLDEN-TAR-BYTES", b"")


class _FakeContainer:
    def exec_run(self, *a, **k):
        return _FakeExec()

    def remove(self, force=False):
        pass


class _FakeContainers:
    def __init__(self):
        self.runs = 0

    def run(self, *a, **k):
        self.runs += 1
        return _FakeContainer()


class _FakeClient:
    def __init__(self):
        self.containers = _FakeContainers()


def _ws(tmp_path):
    ws = tmp_path / "navidrome_navidrome_v0.57.0_v0.58.0"
    ws.mkdir()
    (ws / "metadata.json").write_text('{"repo_src_dirs": ["server", "ui"]}')
    return ws


def test_extract_caches_and_reuses(tmp_path, monkeypatch):
    monkeypatch.setattr(oracle, "_repo_map", lambda: {})  # avoid harness
    ws = _ws(tmp_path)
    cache = tmp_path / "golden-cache"
    c = _FakeClient()

    oracle.extract_selected(c, workspace_root=ws, milestone_ids=["milestone_002"], golden_dir=cache)
    assert c.containers.runs == 1                                   # extracted (miss)
    assert (cache / "milestone_002.tar").read_bytes() == b"GOLDEN-TAR-BYTES"

    # second run: cache hit -> no milestone image acquired
    oracle.extract_selected(c, workspace_root=ws, milestone_ids=["milestone_002"], golden_dir=cache)
    assert c.containers.runs == 1                                   # unchanged (hit)


def test_extract_refresh_forces_reextract(tmp_path, monkeypatch):
    monkeypatch.setattr(oracle, "_repo_map", lambda: {})
    ws = _ws(tmp_path)
    cache = tmp_path / "golden-cache"
    c = _FakeClient()
    oracle.extract_selected(c, workspace_root=ws, milestone_ids=["m"], golden_dir=cache)
    oracle.extract_selected(c, workspace_root=ws, milestone_ids=["m"], golden_dir=cache, refresh=True)
    assert c.containers.runs == 2                                   # refresh re-extracts
