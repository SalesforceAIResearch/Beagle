"""Wrapper-level test for swebench_verified run_full_sweep.sh corpus gate (audit H4).

`--list-green` must NOT return a partial/smoke cache as if it were the full 500-instance
Verified suite (that would green the CI gate off a smoke cache). The default path requires
all 500 present; `--smoke` is the explicit 8-instance subset. Drives the real bash wrapper
against a fabricated shard (no Docker / CP / build).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WRAPPER = (_REPO_ROOT / "xrlenv_plugins" / "benchmarks"
            / "swebench_verified" / "run_full_sweep.sh")


@pytest.fixture(autouse=True)
def _require_bash() -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")


def _make_shard(root: Path, instance_ids: list[str]) -> None:
    """Write SEMANTICALLY-COMPLETE instance dirs (a valid anchor whose id agrees with the dir
    name + matching extracts) — the green-set gate now enumerates only complete dirs (audit
    M13), so a bare ``{}`` anchor would (correctly) be excluded and never green anything."""
    from xrlenv_plugins.benchmarks.swebench_verified.build_cache import _row_to_cache
    shard = root / "swebench-verified"
    for iid in instance_ids:
        row = {"instance_id": iid, "patch": f"diff {iid}", "problem_statement": f"solve {iid}"}
        inst = shard / iid
        inst.mkdir(parents=True, exist_ok=True)
        for name, text in _row_to_cache(row).items():
            (inst / name).write_text(text, encoding="utf-8")


def _list_green(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(root),
        "XRLENV_BENCHMARK_CACHE": str(root),
    }
    return subprocess.run(
        ["bash", str(_WRAPPER), "--list-green", "--skip-build-cache", *extra],
        env=env, capture_output=True, text=True,
    )


def test_list_green_fails_on_partial_cache(tmp_path: Path) -> None:
    # a partial (non-500) cache must NOT be listed as the full suite.
    _make_shard(tmp_path, [f"inst__{i:04d}" for i in range(3)])
    res = _list_green(tmp_path)
    assert res.returncode != 0
    assert "authoritative Verified corpus" in res.stderr


def test_list_green_fails_on_500_fabricated_ids(tmp_path: Path) -> None:
    # audit H4: a cache of 500 ARBITRARY ids has the right COUNT but wrong MEMBERSHIP —
    # it must fail the vendored-manifest membership check, not pass as the Verified corpus.
    _make_shard(tmp_path, [f"fake-{i:04d}" for i in range(500)])
    res = _list_green(tmp_path)
    assert res.returncode != 0
    assert "authoritative Verified corpus" in res.stderr


def test_list_green_fails_on_incomplete_anchors(tmp_path: Path) -> None:
    # audit M13: 500 dirs with the RIGHT ids but BARE {} anchors (no matching extracts, no
    # required fields) are NOT prepared instances. The completeness gate must exclude them so
    # they can't pass membership as the Verified corpus (then fail later loading each row).
    manifest = (_REPO_ROOT / "xrlenv_plugins" / "benchmarks" / "swebench_verified"
                / "verified_instance_ids.txt")
    auth = [ln.strip() for ln in manifest.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    shard = tmp_path / "swebench-verified"
    for iid in auth:
        (shard / iid).mkdir(parents=True, exist_ok=True)
        (shard / iid / "instance.json").write_text("{}", encoding="utf-8")  # bare anchor only
    res = _list_green(tmp_path)
    assert res.returncode != 0
    assert "authoritative Verified corpus" in res.stderr


# The instances run_full_sweep.sh holds out in its EXCLUDE array — the 6 UPSTREAM-ungradeable
# ones (root cause + fix per id in STATUS.md). The 4 flaky psf__requests are intentionally NOT
# excluded (kept IN the green set + documented as flaky). Kept in sync with the bash EXCLUDE
# list; if that changes, update here (and STATUS.md).
_EXPECTED_EXCLUDE = {
    "sphinx-doc__sphinx-8595", "sphinx-doc__sphinx-9711",
    "astropy__astropy-8872", "astropy__astropy-8707", "astropy__astropy-7606",
    "django__django-10097",
}


def test_list_green_passes_on_the_authoritative_500(tmp_path: Path) -> None:
    # the happy path: a cache whose ids ARE the manifest passes membership. The green set is
    # the manifest MINUS the documented EXCLUDE holdouts; INCLUDE union EXCLUDE must reconstruct
    # the full 500-id manifest (audit H4 reconciliation — no fabrication, no silent subset).
    manifest = (_REPO_ROOT / "xrlenv_plugins" / "benchmarks" / "swebench_verified"
                / "verified_instance_ids.txt")
    auth = [ln.strip() for ln in manifest.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    _make_shard(tmp_path, auth)
    res = _list_green(tmp_path)
    assert res.returncode == 0, res.stderr
    ids = [ln for ln in res.stdout.splitlines() if ln and not ln.startswith("==")]
    assert len(auth) == 500
    assert set(ids).isdisjoint(_EXPECTED_EXCLUDE)          # no held-out id leaks into green
    assert set(ids) == set(auth) - _EXPECTED_EXCLUDE       # green = manifest - EXCLUDE
    assert set(ids) | _EXPECTED_EXCLUDE == set(auth)       # H4: INCLUDE union EXCLUDE == manifest


def test_smoke_lists_the_smoke_set(tmp_path: Path) -> None:
    from xrlenv_plugins.benchmarks.swebench_verified.build_cache import SMOKE_INSTANCES

    _make_shard(tmp_path, list(SMOKE_INSTANCES))
    res = _list_green(tmp_path, "--smoke")
    assert res.returncode == 0, res.stderr
    ids = [ln for ln in res.stdout.splitlines() if ln and not ln.startswith("==")]
    assert set(ids) == set(SMOKE_INSTANCES)


def test_smoke_fails_when_a_smoke_instance_is_missing(tmp_path: Path) -> None:
    from xrlenv_plugins.benchmarks.swebench_verified.build_cache import SMOKE_INSTANCES

    _make_shard(tmp_path, list(SMOKE_INSTANCES)[:-1])  # one short
    res = _list_green(tmp_path, "--smoke")
    assert res.returncode != 0
    assert "smoke" in res.stderr.lower()
