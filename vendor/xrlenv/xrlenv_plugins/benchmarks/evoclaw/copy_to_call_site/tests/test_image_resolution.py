"""Unit tests for the EvoClaw image-resolution override (DESIGN.md §5.2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import image_resolution

_PULL_IMAGES_SH = """\
#!/usr/bin/env bash
declare -A REPO_FULL
REPO_FULL[navidrome]="navidrome_navidrome_v0.57.0_v0.58.0"
REPO_FULL[dubbo]="apache_dubbo_dubbo-3.3.3_dubbo-3.3.6"
REPO_FULL[ripgrep]="burntsushi_ripgrep_14.1.1_15.0.0"
REPO_FULL[go-zero]="zeromicro_go-zero_v1.6.0_v1.9.3"
"""


@pytest.fixture()
def evoclaw_root(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "pull_images.sh").write_text(_PULL_IMAGES_SH)
    return tmp_path


def test_load_repo_short_map(evoclaw_root):
    m = image_resolution.load_repo_short_map(evoclaw_root)
    assert m["navidrome_navidrome_v0.57.0_v0.58.0"] == "navidrome"
    # short name is NOT derivable from the full name — this is why we parse it
    assert m["apache_dubbo_dubbo-3.3.3_dubbo-3.3.6"] == "dubbo"
    assert m["burntsushi_ripgrep_14.1.1_15.0.0"] == "ripgrep"


def test_load_repo_short_map_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        image_resolution.load_repo_short_map(tmp_path)


def test_dockerhub_ref_default(evoclaw_root):
    m = image_resolution.load_repo_short_map(evoclaw_root)
    ref = image_resolution.dockerhub_ref(
        "navidrome_navidrome_v0.57.0_v0.58.0/milestone_001", m
    )
    assert ref == "hyd2apse/navidrome:milestone_001-v0.9"


def test_dockerhub_ref_dubbo_nontrivial_mapping(evoclaw_root):
    m = image_resolution.load_repo_short_map(evoclaw_root)
    ref = image_resolution.dockerhub_ref(
        "apache_dubbo_dubbo-3.3.3_dubbo-3.3.6/milestone_002", m
    )
    assert ref == "hyd2apse/dubbo:milestone_002-v0.9"


def test_dockerhub_ref_image_tag_is_a_flag(evoclaw_root):
    # tag comes from the --image-tag flag (a kwarg), NOT the environment
    m = image_resolution.load_repo_short_map(evoclaw_root)
    ref = image_resolution.dockerhub_ref(
        "navidrome_navidrome_v0.57.0_v0.58.0/milestone_001", m, image_tag="v1.0"
    )
    assert ref == "hyd2apse/navidrome:milestone_001-v1.0"


def test_dockerhub_ref_image_registry_is_a_flag(evoclaw_root):
    # explicit mirror-host prefix comes from --image-registry (a kwarg), not env
    m = image_resolution.load_repo_short_map(evoclaw_root)
    ref = image_resolution.dockerhub_ref(
        "navidrome_navidrome_v0.57.0_v0.58.0/milestone_001", m,
        image_tag="v1.0", image_registry="mirror.local:5000",
    )
    assert ref == "mirror.local:5000/hyd2apse/navidrome:milestone_001-v1.0"


def test_dockerhub_ref_unknown_repo_returns_none(evoclaw_root):
    m = image_resolution.load_repo_short_map(evoclaw_root)
    assert image_resolution.dockerhub_ref("some_other_repo/milestone_001", m) is None


# --- go-zero base-image redirect: the one image knob kept as env
#     (EVOCLAW_GOZERO_BASE_IMAGE), REQUIRED (no default) ---
_GZ = "zeromicro_go-zero_v1.6.0_v1.9.3"


def test_gozero_base_unset_fails_loud(evoclaw_root, monkeypatch):
    monkeypatch.delenv("EVOCLAW_GOZERO_BASE_IMAGE", raising=False)  # unset -> fail loud
    m = image_resolution.load_repo_short_map(evoclaw_root)
    with pytest.raises(SystemExit):
        image_resolution.dockerhub_ref(f"{_GZ}/base", m)


def test_gozero_base_env_override(evoclaw_root, monkeypatch):
    monkeypatch.setenv("EVOCLAW_GOZERO_BASE_IMAGE", "myreg:5000/go-zero:fixed")
    m = image_resolution.load_repo_short_map(evoclaw_root)
    assert image_resolution.dockerhub_ref(f"{_GZ}/base", m) == "myreg:5000/go-zero:fixed"


def test_gozero_redirect_disabled_when_env_empty(evoclaw_root, monkeypatch):
    monkeypatch.setenv("EVOCLAW_GOZERO_BASE_IMAGE", "")  # empty -> upstream ref
    m = image_resolution.load_repo_short_map(evoclaw_root)
    assert image_resolution.dockerhub_ref(f"{_GZ}/base", m) == "hyd2apse/go-zero:base-v0.9"


def test_gozero_redirect_only_affects_base_not_milestones(evoclaw_root, monkeypatch):
    monkeypatch.delenv("EVOCLAW_GOZERO_BASE_IMAGE", raising=False)
    m = image_resolution.load_repo_short_map(evoclaw_root)
    # milestone ref is untouched — only the base image is broken upstream
    assert image_resolution.dockerhub_ref(f"{_GZ}/M005", m) == "hyd2apse/go-zero:M005-v0.9"


def test_nongozero_base_unaffected_by_redirect(evoclaw_root, monkeypatch):
    monkeypatch.delenv("EVOCLAW_GOZERO_BASE_IMAGE", raising=False)
    m = image_resolution.load_repo_short_map(evoclaw_root)
    assert image_resolution.dockerhub_ref("navidrome_navidrome_v0.57.0_v0.58.0/base", m) \
        == "hyd2apse/navidrome:base-v0.9"
