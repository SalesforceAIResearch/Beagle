"""Offline tests for the swebench-pro cache builder: pure renderers + the writer on a fake upstream kit."""
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from xrlenv_plugins.benchmarks.swebench_pro import build_cache as bc

ROW = {
    "repo": "NodeBB/NodeBB", "instance_id": "instance_NodeBB__NodeBB-0499-vnan", "base_commit": "1e137b07052bc3ea0da44ed201702c94055b8ad2",
    "patch": "diff --git a/src/a.js b/src/a.js\n--- a/src/a.js\n+++ b/src/a.js\n@@ -1 +1,2 @@\n x\n+y\n", "test_patch": "",
    "problem_statement": "Keys API returns wrong values\r\nwith CRLF text", "requirements": "mget must return null for missing keys",
    "interface": "db.mget(keys: string[]) -> Array", "repo_language": "js",
    "fail_to_pass": json.dumps(["test/database.js | Test database keys should return multiple keys"]),
    "pass_to_pass": json.dumps(["test/database.js | Test database keys existing"]),
    "issue_specificity": "[]", "issue_categories": '["back_end_knowledge"]',
    "before_repo_set_cmd": "git reset --hard 1e137b0\ngit checkout 1e137b0\ngit checkout 04998908 -- test/database.js test/database/keys.js",
    "selected_test_files_to_run": json.dumps(["test/database.js", "test/database/keys.js"]),
    "dockerhub_tag": "nodebb.nodebb-NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5",
}


def _kit(tmp_path: Path, iid: str) -> Path:
    h = tmp_path / "harness"
    (h / "run_scripts" / iid).mkdir(parents=True)
    (h / "dockerfiles" / "base_dockerfile" / iid).mkdir(parents=True)
    (h / "dockerfiles" / "instance_dockerfile" / iid).mkdir(parents=True)
    (h / "run_scripts" / iid / "run_script.sh").write_text("#!/bin/bash\necho tests $@\n")
    (h / "run_scripts" / iid / "parser.py").write_text("import json,sys; json.dump({'tests': []}, open(sys.argv[3], 'w'))\n")
    (h / "dockerfiles" / "base_dockerfile" / iid / "Dockerfile").write_text("FROM ubuntu\nENV NODE_ENV=test\nRUN true\n")
    (h / "dockerfiles" / "instance_dockerfile" / iid / "Dockerfile").write_text("FROM base\nENV PYTEST_ADDOPTS=\"--tb=short\"\nENV UV_HTTP_TIMEOUT=60\n")
    return h


def test_task_toml_carries_the_hub_image_and_sizing():
    toml = tomllib.loads(bc.render_task_toml(ROW, kept=True))
    assert toml["environment"]["docker_image"] == "jefzda/sweap-images:nodebb.nodebb-NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5"
    assert toml["environment"]["cpus"] == 4 and toml["environment"]["memory_mb"] == 12288       # js sizing; NodeBB is a heavy repo -> 12288
    assert toml["verifier"]["timeout_sec"] == 2400.0 and toml["agent"]["timeout_sec"] == 5400.0
    assert toml["metadata"]["instance_id"] == ROW["instance_id"] and toml["metadata"]["filter_kept"] is True
    assert toml["task"]["name"] == "swebench-pro/" + ROW["instance_id"]
    assert toml["environment"]["env"]["XRLENV_KEEPALIVE_ENTRYPOINT"] == "1"        # images have ENTRYPOINT /bin/bash
    from xrlenv_plugins.harbor.environment import _keepalive_argv
    assert _keepalive_argv(toml["environment"]["env"], False) == (["sleep"], ["infinity"])
    heavy = dict(ROW, repo="protonmail/webclients", repo_language="ts")
    t2 = tomllib.loads(bc.render_task_toml(heavy))
    assert t2["environment"]["memory_mb"] == 16384 and t2["verifier"]["timeout_sec"] == 3600.0
    t3 = tomllib.loads(bc.render_task_toml(dict(ROW, repo="element-hq/element-web", repo_language="ts")))
    assert t3["environment"]["memory_mb"] == 32768                                             # jest OOM at 16 GiB (2026-08-27)
    with pytest.raises(SystemExit):
        bc.image_ref(dict(ROW, dockerhub_tag=""))


def test_test_sh_mirrors_upstream_entry_script():
    sh = bc.render_test_sh(ROW)
    assert "source /tests/env.sh" in sh
    assert "git reset --hard 1e137b07052bc3ea0da44ed201702c94055b8ad2" in sh and "git checkout 1e137b07052bc3ea0da44ed201702c94055b8ad2" in sh
    assert "git checkout 04998908 -- test/database.js test/database/keys.js" in sh              # only the LAST line of before_repo_set_cmd
    assert "git reset --hard 1e137b0\n" not in sh
    assert "bash /tests/run_script.sh test/database.js,test/database/keys.js" in sh                # ONE comma-joined argument, as upstream
    assert "git diff --cached --binary" in sh and "git apply -v /logs/verifier/model.patch" in sh
    assert "/tests/grade.py /logs/verifier/output.json /tests/f2p.json /tests/p2p.json /logs/verifier/reward.json > /logs/verifier/reward.txt" in sh


def test_env_exports_mirror_upstream_quirks():
    env = bc.dockerfile_env_exports("FROM x\nENV A=1\nRUN y\n", "ENV B C\nENV D=\"e f\"\n")
    assert env.splitlines()[2:] == ["export A=1", "export B C", 'export D="e f"']         # verbatim ENV -> export, quirks included


def test_json_list_accepts_json_and_python_literals():
    assert bc._json_list('["a", "b"]') == ["a", "b"] and bc._json_list("['a', \"it's\"]") == ["a", "it's"] and bc._json_list("") == []


def test_grade_rule_matches_upstream(tmp_path: Path):
    out = tmp_path / "output.json"
    f2p = tmp_path / "f2p.json"
    p2p = tmp_path / "p2p.json"
    rew = tmp_path / "reward.json"
    grade = tmp_path / "grade.py"
    grade.write_text(bc.GRADE_PY)
    f2p.write_text(json.dumps(["t1", "t2"]))
    p2p.write_text(json.dumps(["t3"]))
    out.write_text(json.dumps({"tests": [{"name": "t1", "status": "PASSED"}, {"name": "t2", "status": "PASSED"}, {"name": "t3", "status": "PASSED"}, {"name": "t9", "status": "FAILED"}]}))
    r = subprocess.run([sys.executable, str(grade), str(out), str(f2p), str(p2p), str(rew)], capture_output=True, text=True)
    assert r.stdout.strip() == "1" and json.loads(rew.read_text())["resolved"] == 1
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in json.loads(rew.read_text()).values())   # harbor: dict[str, float|int]
    out.write_text(json.dumps({"tests": [{"name": "t1", "status": "PASSED"}, {"name": "t2", "status": "FAILED"}, {"name": "t3", "status": "PASSED"}]}))
    r = subprocess.run([sys.executable, str(grade), str(out), str(f2p), str(p2p), str(rew)], capture_output=True, text=True)
    j = json.loads(rew.read_text())
    d = json.loads((tmp_path / "grade_details.json").read_text())
    assert r.stdout.strip() == "0" and j["f2p_passed"] == 1 and d["missing_f2p"] == ["t2"]
    out.write_text("not json")
    r = subprocess.run([sys.executable, str(grade), str(out), str(f2p), str(p2p), str(rew)], capture_output=True, text=True)
    assert r.stdout.strip() == "0" and json.loads((tmp_path / "grade_details.json").read_text())["error"]


def test_writer_is_complete_and_idempotent(tmp_path: Path):
    iid = ROW["instance_id"]
    h = _kit(tmp_path, iid)
    shard = tmp_path / "cache" / "swebench-pro"
    bc.write_task(ROW, shard / iid, h, kept=True)
    assert bc.is_complete(shard / iid, iid) and not bc.is_complete(shard / iid, "other")
    assert (shard / iid / "instruction.md").read_bytes().count(b"\r\n") == 1                      # dataset bytes kept verbatim
    assert (shard / iid / "tests" / "test.sh").stat().st_mode & 0o111 and (shard / iid / "solution" / "solve.sh").stat().st_mode & 0o111
    assert json.loads((shard / iid / "tests" / "f2p.json").read_text()) == json.loads(ROW["fail_to_pass"])
    assert (shard / iid / "environment" / "Dockerfile").read_text() == "FROM " + bc.image_ref(ROW) + "\n"
    assert "export NODE_ENV=test" in (shard / iid / "tests" / "env.sh").read_text()
    # a complete dir whose kit-rendered files predate the renderer is refreshed in place (grading fixes propagate)
    assert bc.refresh_kit_files(ROW, shard / iid) is False
    (shard / iid / "tests" / "grade.py").write_text("# old grading rule\n")
    (shard / iid / "tests" / "test.sh").write_text("#!/bin/bash\necho old\n")
    assert bc.refresh_kit_files(ROW, shard / iid) is True
    assert (shard / iid / "tests" / "grade.py").read_text() == bc.GRADE_PY and (shard / iid / "tests" / "test.sh").read_text() == bc.render_test_sh(ROW)
    assert (shard / iid / "tests" / "test.sh").stat().st_mode & 0o111 and bc.refresh_kit_files(ROW, shard / iid) is False
    (shard / iid / "instance.json").unlink()
    assert not bc.is_complete(shard / iid, iid)


def test_select_rows_manifest_and_smoke(tmp_path: Path):
    rows = [dict(ROW, instance_id=f"instance_x-{i}") for i in range(10)]
    assert [r["instance_id"] for r in bc.select_rows(rows, all_=False, smoke=True, ids_file=None, instances=None)] == [f"instance_x-{i}" for i in range(8)]
    f = tmp_path / "ids.txt"
    f.write_text("# comment\ninstance_x-3\n\ninstance_x-1\n")
    assert [r["instance_id"] for r in bc.select_rows(rows, all_=False, smoke=False, ids_file=f, instances=None)] == ["instance_x-3", "instance_x-1"]
    with pytest.raises(SystemExit):
        bc.select_rows(rows, all_=False, smoke=False, ids_file=None, instances="instance_x-99")
    ff = tmp_path / "filtered_instance_ids.txt"
    ff.write_text("instance_x-5\n")
    monkey = bc.FILTERED_IDS
    bc.FILTERED_IDS = ff
    try:
        assert [r["instance_id"] for r in bc.select_rows(rows, all_=False, smoke=False, ids_file=None, instances=None, filtered=True)] == ["instance_x-5"]
    finally:
        bc.FILTERED_IDS = monkey
    with pytest.raises(SystemExit):                                                        # a selection is always explicit
        bc.select_rows(rows, all_=False, smoke=False, ids_file=None, instances=None)
    with pytest.raises(SystemExit):
        bc.safe_instance_id("../victim")


def test_a_named_input_is_honoured_verbatim(tmp_path: Path, monkeypatch):
    """An explicit flag or env var wins over fetching, and resolves file-or-snapshot-directory."""
    monkeypatch.setattr(bc, "fetch_parquet", _never_fetch)
    snap = tmp_path / "snapshot" / "data"
    snap.mkdir(parents=True)
    (snap / "test-00000-of-00001.parquet").write_bytes(b"")
    monkeypatch.setenv(bc.PARQUET_ENV, str(tmp_path / "snapshot"))
    assert bc.parquet_path() == snap / "test-00000-of-00001.parquet"                      # a snapshot directory resolves to its parquet
    assert bc.parquet_path(str(snap / "test-00000-of-00001.parquet")).name == "test-00000-of-00001.parquet"

    harness = tmp_path / "kit"
    (harness / "run_scripts").mkdir(parents=True)
    (harness / "dockerfiles").mkdir()
    monkeypatch.setenv(bc.HARNESS_ENV, str(harness))
    monkeypatch.setattr(bc, "fetch_harness", _never_fetch)
    assert bc.harness_dir() == harness


def _never_fetch(*a, **k):
    raise AssertionError("must not fetch when an input is named")


def test_a_bad_named_input_fails_loud_and_does_not_download(tmp_path: Path, monkeypatch):
    """THE precedence rule. An operator who named a location meant it, so a typo must be reported —
    silently downloading over it would evaluate a DIFFERENT corpus than they asked for."""
    monkeypatch.setattr(bc, "fetch_parquet", _never_fetch)
    monkeypatch.setattr(bc, "fetch_harness", _never_fetch)

    monkeypatch.setenv(bc.PARQUET_ENV, str(tmp_path / "empty"))
    with pytest.raises(SystemExit, match=bc.PARQUET_ENV):
        bc.parquet_path()
    with pytest.raises(SystemExit, match=bc.PARQUET_ENV):
        bc.parquet_path(str(tmp_path / "empty"))                                          # via the flag too

    not_a_kit = tmp_path / "not-a-kit"
    not_a_kit.mkdir()
    monkeypatch.setenv(bc.HARNESS_ENV, str(not_a_kit))
    with pytest.raises(SystemExit, match="run_scripts"):
        bc.harness_dir()


def test_unset_inputs_are_fetched_not_refused(tmp_path: Path, monkeypatch):
    """The kit is self-contained: with NOTHING set, both inputs are provisioned rather than
    demanded of the operator. (Regression — these used to raise SystemExit, which made
    `run_benchmarks.py --profile ci` fail at planning on any box without the env vars.)"""
    monkeypatch.delenv(bc.PARQUET_ENV, raising=False)
    monkeypatch.delenv(bc.HARNESS_ENV, raising=False)
    parquet = tmp_path / "test-00000-of-00001.parquet"
    parquet.write_bytes(b"")
    kit = tmp_path / "kit"
    monkeypatch.setattr(bc, "fetch_parquet", lambda: parquet)
    monkeypatch.setattr(bc, "fetch_harness", lambda dest=None: kit)
    assert bc.parquet_path() == parquet
    assert bc.harness_dir() == kit


def test_fetch_harness_reuses_an_existing_clone(tmp_path: Path, monkeypatch):
    """Cached under the cache ROOT, not a temp dir, so it is cloned once and shared by every entry
    point. A second call must not shell out to git at all."""
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(tmp_path))
    d = tmp_path / ".upstream" / "SWE-bench_Pro-os"
    (d / "run_scripts").mkdir(parents=True)
    (d / "dockerfiles").mkdir()

    import subprocess as sp

    def boom(*_a, **_k):
        raise AssertionError("must not re-clone an existing checkout")

    monkeypatch.setattr(sp, "run", boom)
    assert bc.fetch_harness() == d


def test_fetch_harness_clears_a_partial_clone(tmp_path: Path, monkeypatch):
    """A clone killed part-way leaves a dir that is neither a valid kit nor an empty target — git
    would refuse it forever. It must be cleared rather than wedging the kit."""
    import subprocess as sp
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(tmp_path))
    d = tmp_path / ".upstream" / "SWE-bench_Pro-os"
    (d / "leftover").mkdir(parents=True)          # partial: no run_scripts/, no dockerfiles/

    def fake_clone(cmd, **k):
        assert not Path(cmd[-1]).exists(), "target must be cleared before git clone"
        (Path(cmd[-1]) / "run_scripts").mkdir(parents=True)
        (Path(cmd[-1]) / "dockerfiles").mkdir()
        return sp.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sp, "run", fake_clone)
    assert bc.fetch_harness() == d
    assert not (d / "leftover").exists()
    assert bc.harness_dir(str(_kit(tmp_path, "instance_k"))).name == "harness"


def test_grade_tolerates_dataset_name_mangling(tmp_path: Path):
    """Names cut at an embedded quote / trailing-space differences (dataset artifacts) still match; anything else is exact."""
    out = tmp_path / "output.json"
    f2p = tmp_path / "f2p.json"
    p2p = tmp_path / "p2p.json"
    rew = tmp_path / "reward.json"
    grade = tmp_path / "grade.py"
    grade.write_text(bc.GRADE_PY)
    f2p.write_text(json.dumps(['a | ACP default "day', "b | big arrays (length > 100)", "c | plain"]))
    p2p.write_text("[]")
    out.write_text(json.dumps({"tests": [{"name": 'a | ACP default "day"', "status": "PASSED"}, {"name": "b | big arrays (length > 100) ", "status": "PASSED"},
                                         {"name": "c | plain", "status": "PASSED"}]}))
    r = subprocess.run([sys.executable, str(grade), str(out), str(f2p), str(p2p), str(rew)], capture_output=True, text=True)
    assert r.stdout.strip() == "1" and len(json.loads((tmp_path / "grade_details.json").read_text())["lenient_matches"]) == 2
    out.write_text(json.dumps({"tests": [{"name": "c | plain-extra", "status": "PASSED"}]}))       # a balanced name never prefix-matches
    f2p.write_text(json.dumps(["c | plain"]))
    r = subprocess.run([sys.executable, str(grade), str(out), str(f2p), str(p2p), str(rew)], capture_output=True, text=True)
    assert r.stdout.strip() == "0"


def test_shard_dir_resolves_both_layouts(tmp_path):
    """Canonical is <root>/swebench-pro/<id>/; the shared cluster cache nests the identical dirs under
    swebench-pro/golden_patches/. Flat wins when populated; an empty root stays canonical so a populate
    never writes into the nested layout by accident."""
    from xrlenv_plugins.benchmarks.swebench_pro.build_cache import GOLDEN_SUBDIR, SHARD, shard_dir

    def task(d):
        (d / "i1").mkdir(parents=True)
        (d / "i1" / "task.toml").write_text("[task]\n")
        return d

    empty = tmp_path / "empty"
    assert shard_dir(empty) == empty / SHARD                      # nothing populated -> canonical
    nested_only = tmp_path / "shared"
    task(nested_only / SHARD / GOLDEN_SUBDIR)
    assert shard_dir(nested_only) == nested_only / SHARD / GOLDEN_SUBDIR
    both = tmp_path / "both"
    task(both / SHARD)
    task(both / SHARD / GOLDEN_SUBDIR)
    assert shard_dir(both) == both / SHARD                        # flat wins
    flat_only = tmp_path / "flat"
    task(flat_only / SHARD)
    assert shard_dir(flat_only) == flat_only / SHARD


def test_sweep_script_mirrors_the_shard_rule():
    """run_full_sweep.sh gates the green set in bash, so it must apply the same fallback as shard_dir()."""
    from pathlib import Path
    sh = (Path(__file__).resolve().parents[1] / "run_full_sweep.sh").read_text()
    assert 'SHARD="$SHARD/golden_patches"' in sh
    assert 'compgen -G "$SHARD"/*/task.toml' in sh and 'compgen -G "$SHARD/golden_patches"/*/task.toml' in sh
