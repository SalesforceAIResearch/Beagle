"""The three shipped configurations — full (731) / filtered (478) / subset-100 — agree with each other:
manifests, image plans, sweep scripts and the README, and nothing in the kit points at an absolute data path."""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml
from xrlenv_plugins.benchmarks.swebench_pro import build_cache as bc

KIT = Path(__file__).resolve().parents[1]
SCRIPTS = KIT / "scripts"
# the kit root holds the full-corpus path only; everything derived (filtered / subset-100 manifests, plans, pinned
# wrappers, the smoke wrapper, the generators) lives under scripts/
ROOT_FILES = {"README.md", "STATUS.md", "build_cache.py", "build_plan_full.yaml", "run_full_sweep.sh", "run_oracle_sweep.py"}
ROOT_DIRS = {"scripts", "tests"}
SCRIPT_FILES = {"README.md", "build_plan_filtered.yaml", "build_plan_gen.py", "build_plan_subset_100.yaml", "filter_report.json",
                "filtered_instance_ids.txt", "run_100_subset_sweep.sh", "run_filtered_sweep.sh", "run_smoke_one.sh",
                "sample_subset.py", "subset_100.json", "subset_100_instance_ids.txt"}


def _ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def _plan(name: str) -> dict:
    return yaml.safe_load(((KIT if name == "build_plan_full.yaml" else SCRIPTS) / name).read_text())


def _plan_ids(plan: dict) -> list[str]:
    return [e["labels"]["xrlenv.instance_id"] for e in plan["entries"]]


def _sizes(plan: dict) -> dict[str, int]:
    return {e["image_ref"]: e["placement"]["size_hint_bytes"] for e in plan["entries"]}


def test_full_plan_covers_the_corpus():
    full = _plan("build_plan_full.yaml")
    ids = _plan_ids(full)
    assert full["name"] == "swebench-pro-full-731" and len(ids) == 731 and len(set(ids)) == 731
    assert all(e["context_source"] == {"type": "registry"} and e["placement"]["size_hint_source"] == "registry-probe" for e in full["entries"])


def test_filtered_configuration():
    kept = _ids(SCRIPTS / "filtered_instance_ids.txt")
    assert len(kept) == 478 and len(set(kept)) == 478 and set(kept) <= set(_plan_ids(_plan("build_plan_full.yaml")))
    plan = _plan("build_plan_filtered.yaml")
    assert plan["name"] == "swebench-pro-filtered-478" and _plan_ids(plan) == kept                 # plan order == manifest order


def test_subset_100_configuration():
    ids = _ids(SCRIPTS / "subset_100_instance_ids.txt")
    kept = set(_ids(SCRIPTS / "filtered_instance_ids.txt"))
    assert len(ids) == 100 and len(set(ids)) == 100 and set(ids) <= kept                          # 100 distinct ids, all from the filtered set
    report = json.loads((SCRIPTS / "subset_100.json").read_text())
    assert report["total"] == 100 and report["policy"] == "random" and report["seed"] == 0 and report["n_kept"] == len(kept)
    assert not os.path.isabs(report["kept_manifest"])
    picks = report["picks"]
    assert [p["instance_id"] for p in picks] == ids                                                 # report order == manifest order
    per_repo = Counter(p["repo"] for p in picks)
    assert len(per_repo) == 11 and min(per_repo.values()) >= 1 and max(per_repo.values()) <= 13      # every repo, none dominating
    plan = _plan("build_plan_subset_100.yaml")
    assert plan["name"] == "swebench-pro-subset-100" and _plan_ids(plan) == ids
    assert {e["image_ref"] for e in plan["entries"]} == {p["image_ref"] for p in picks}


def test_derived_plans_carry_the_full_plans_sizes():
    full = _sizes(_plan("build_plan_full.yaml"))
    for name in ("build_plan_filtered.yaml", "build_plan_subset_100.yaml"):
        sizes = _sizes(_plan(name))
        assert sizes and all(full.get(ref) == size for ref, size in sizes.items()), name


def test_build_cache_selects_each_configuration_in_manifest_order():
    kept = _ids(SCRIPTS / "filtered_instance_ids.txt")
    sub = _ids(SCRIPTS / "subset_100_instance_ids.txt")
    rows = [{"instance_id": i} for i in reversed(kept)]
    assert [r["instance_id"] for r in bc.select_rows(rows, all_=False, smoke=False, ids_file=None, instances=None, filtered=True)] == kept
    assert [r["instance_id"] for r in bc.select_rows(rows, all_=False, smoke=False, ids_file=None, instances=None, subset_100=True)] == sub
    assert len(bc.select_rows(rows, all_=True, smoke=False, ids_file=None, instances=None)) == len(kept)
    p = bc.build_parser()
    assert p.parse_args(["--all"]).all and p.parse_args(["--filtered"]).filtered and p.parse_args(["--subset-100"]).subset_100
    with pytest.raises(SystemExit):
        p.parse_args([])                                                                            # a selection is required
    with pytest.raises(SystemExit):
        p.parse_args(["--all", "--filtered"])


def test_sweep_scripts_and_readmes_name_the_configurations():
    full = (KIT / "run_full_sweep.sh").read_text()
    assert "--filtered" in full and "--subset-100" in full and "--repo-unique" not in full
    for wrapper, flag in (("run_filtered_sweep.sh", "--filtered"), ("run_100_subset_sweep.sh", "--subset-100")):
        text = (SCRIPTS / wrapper).read_text()
        assert f'"$HERE/../run_full_sweep.sh" {flag} "$@"' in text and os.access(SCRIPTS / wrapper, os.X_OK), wrapper
    smoke = (SCRIPTS / "run_smoke_one.sh").read_text()                                                 # one task through the same pipeline
    assert '"$HERE/../run_full_sweep.sh" --instances "$INSTANCE" --max-workers 1' in smoke and "--instances)" in full and os.access(SCRIPTS / "run_smoke_one.sh", os.X_OK)
    readme = (KIT / "README.md").read_text()                                                     # root: the full corpus only
    for name in ("build_cache.py --all", "run_full_sweep.sh", "build_plan_full.yaml", "scripts/README.md",
                 "SWEBENCH_PRO_PARQUET", "SWEBENCH_PRO_HARNESS", "XRLENV_BENCHMARK_CACHE"):
        assert name in readme, name
    for name in ("run_filtered_sweep.sh", "run_100_subset_sweep.sh", "run_smoke_one.sh", "--filtered", "--subset-100"):
        assert name not in readme, f"{name} belongs in scripts/README.md, not the root README"
    partitions = (SCRIPTS / "README.md").read_text()                                            # scripts/: the partitions
    for name in ("run_filtered_sweep.sh", "run_100_subset_sweep.sh", "run_smoke_one.sh", "build_plan_filtered.yaml",
                 "build_plan_subset_100.yaml", "filtered_instance_ids.txt", "subset_100_instance_ids.txt", "filter_report.json",
                 "subset_100.json", "sample_subset.py", "build_plan_gen.py", "--filtered", "--subset-100"):
        assert name in partitions, name
    status = (KIT / "STATUS.md").read_text()                                                     # root STATUS: the full sweep
    assert "run_full_sweep.sh" in status and "scripts/README.md" in status
    for name in ("run_filtered_sweep.sh", "run_100_subset_sweep.sh", "run_smoke_one.sh"):
        assert name not in status, f"{name} belongs in scripts/README.md, not STATUS.md"


def test_kit_layout():
    """Root = the full-corpus path; scripts/ = the derived configurations, the smoke wrapper and the generators."""
    root = {p.name for p in KIT.iterdir() if p.name != "__pycache__"}
    assert root == ROOT_FILES | ROOT_DIRS, root ^ (ROOT_FILES | ROOT_DIRS)
    scripts = {p.name for p in SCRIPTS.iterdir() if p.name != "__pycache__"}
    assert scripts == SCRIPT_FILES, scripts ^ SCRIPT_FILES


def test_kit_names_no_absolute_data_paths():
    """Inputs are env/flags only (SWEBENCH_PRO_PARQUET, SWEBENCH_PRO_HARNESS, XRLENV_BENCHMARK_CACHE): no home-directory
    or site-specific path is baked into any shipped script, doc or manifest."""
    pattern = re.compile(r"(^|[\s`'\"=(])(~/|/fsx/|/home/)")
    for path in sorted([*KIT.iterdir(), *SCRIPTS.iterdir()]):
        if path.suffix in (".py", ".sh", ".md", ".txt"):
            assert not pattern.search(path.read_text()), path.name
    assert not os.path.isabs(json.loads((SCRIPTS / "subset_100.json").read_text())["kept_manifest"])
