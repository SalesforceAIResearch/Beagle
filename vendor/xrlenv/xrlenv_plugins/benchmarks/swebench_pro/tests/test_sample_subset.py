"""Offline tests for the repo-balanced sampler and the plan generator's manifest selection."""
from __future__ import annotations

from pathlib import Path

import pytest
from xrlenv_plugins.benchmarks.swebench_pro.scripts import build_plan_gen as bpg
from xrlenv_plugins.benchmarks.swebench_pro.scripts import sample_subset as ss

ROWS = [
    {"instance_id": "instance_a__x-1", "repo": "a/x", "repo_language": "go", "dockerhub_tag": "t1"},
    {"instance_id": "instance_a__x-2", "repo": "a/x", "repo_language": "go", "dockerhub_tag": "t2"},
    {"instance_id": "instance_b__y-1", "repo": "b/y", "repo_language": "python", "dockerhub_tag": "t3"},
    {"instance_id": "instance_b__y-2", "repo": "b/y", "repo_language": "python", "dockerhub_tag": "t4"},
    {"instance_id": "instance_c__z-1", "repo": "c/z", "repo_language": "js", "dockerhub_tag": "t5"},
]
SIZES = {"jefzda/sweap-images:t1": 900, "jefzda/sweap-images:t2": 100, "jefzda/sweap-images:t3": 500}   # t4/t5 unknown


def test_one_per_repo_from_kept_only_and_policies():
    kept = ["instance_a__x-1", "instance_a__x-2", "instance_b__y-2", "instance_c__z-1"]      # b/y-1 not kept
    picks = ss.select_subset(ROWS, kept, policy="smallest-image", sizes=SIZES)
    assert [p["repo"] for p in picks] == ["a/x", "b/y", "c/z"]                           # every repo exactly once, dataset order
    assert [p["instance_id"] for p in picks] == ["instance_a__x-2", "instance_b__y-2", "instance_c__z-1"]   # smallest image; unknown size sorts last
    assert picks[0]["kept_in_repo"] == 2 and picks[0]["size_hint_bytes"] == 100 and picks[1]["size_hint_bytes"] is None
    first = ss.select_subset(ROWS, kept, policy="first")
    assert [p["instance_id"] for p in first] == ["instance_a__x-1", "instance_b__y-2", "instance_c__z-1"]
    r1 = ss.select_subset(ROWS, kept, policy="random", seed=3)
    r2 = ss.select_subset(ROWS, kept, policy="random", seed=3)
    assert r1 == r2 and all(p["instance_id"] in kept for p in r1)                       # deterministic, kept-only
    with pytest.raises(ValueError):
        ss.select_subset(ROWS, kept, policy="bogus")
    with pytest.raises(SystemExit):
        ss.select_subset(ROWS, ["instance_zzz"])


def test_load_sizes_reads_only_registry_probed(tmp_path: Path):
    plan = tmp_path / "plan.yaml"
    plan.write_text("entries:\n- image_ref: jefzda/sweap-images:t1\n  placement: {size_hint_bytes: 42, size_hint_source: registry-probe}\n"
                    "- image_ref: jefzda/sweap-images:t2\n  placement: {size_hint_bytes: 7, size_hint_source: heuristic}\n")
    assert ss.load_sizes(plan) == {"jefzda/sweap-images:t1": 42} and ss.load_sizes(tmp_path / "missing.yaml") == {}


def test_plan_gen_selects_from_manifest_files(tmp_path: Path):
    rows = [{"instance_id": r["instance_id"], "dockerhub_tag": r["dockerhub_tag"]} for r in ROWS]
    f = tmp_path / "ids.txt"
    f.write_text("# picks\ninstance_c__z-1\ninstance_a__x-2\n")
    sel = bpg._select_instances(rows, all_=False, smoke=False, instances=None, ids_file=f)
    assert [r["instance_id"] for r in sel] == ["instance_c__z-1", "instance_a__x-2"]           # manifest order kept
    with pytest.raises(SystemExit):
        bpg._select_instances(rows, all_=False, smoke=False, instances=None, ids_file=tmp_path / "nope.txt", subset_100=True)
    assert bpg.FILTERED_IDS.is_file() and bpg.SUBSET_100_IDS.is_file()                        # the shipped manifests
    plan = bpg.generate_plan(sel, probe_sizes=False, plan_label="subset")
    assert plan["name"] == "swebench-pro-subset-2" and [e["image_ref"] for e in plan["entries"]] == ["jefzda/sweap-images:t5", "jefzda/sweap-images:t2"]
    assert bpg.generate_plan(sel, probe_sizes=False, plan_label="full")["name"] == "swebench-pro-full-2"
    assert bpg.generate_plan(sel, probe_sizes=False, plan_label="filtered")["name"] == "swebench-pro-filtered-2"


def test_per_repo_cap_and_proportional_total():
    rows = [{"instance_id": f"instance_{r}-{i}", "repo": r, "repo_language": "go", "dockerhub_tag": f"{r}{i}"} for r, n in (("a/x", 6), ("b/y", 3), ("c/z", 1)) for i in range(n)]
    kept = [r["instance_id"] for r in rows]
    cap = ss.select_subset(rows, kept, policy="first", per_repo=2)
    assert [p["instance_id"] for p in cap] == ["instance_a/x-0", "instance_a/x-1", "instance_b/y-0", "instance_b/y-1", "instance_c/z-0"]   # min(N, kept)
    tot = ss.select_subset(rows, kept, policy="first", total=5)
    counts = {}
    for p in tot:
        counts[p["repo"]] = counts.get(p["repo"], 0) + 1
    assert sum(counts.values()) == 5 and counts["c/z"] == 1 and counts["a/x"] >= counts["b/y"] >= 1      # proportional, every repo ≥ 1
    assert len(ss.select_subset(rows, kept, policy="first", total=999)) == 10                     # capped by availability
    rnd = ss.select_subset(rows, kept, policy="random", per_repo=3, seed=1)
    assert len(rnd) == 3 + 3 + 1 and len({p["instance_id"] for p in rnd}) == 7                        # no duplicates
