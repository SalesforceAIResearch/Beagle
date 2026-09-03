"""Tests for scripts/generate_eval_configs.py — the config shape lives in the script (build_config);
it constructs one runnable config per (onboarded agent × benchmark), source filled from the manifest
matched on version."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_eval_configs.py"
_spec = importlib.util.spec_from_file_location("generate_eval_configs", _PATH)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)  # type: ignore[union-attr]

#: The join key is ``version``, so tests take it FROM the matrix rather than restating it — a
#: literal here goes stale the moment an agent is re-onboarded at a new version.
_MONET_VERSION = str(gen.AGENTS["monet"]["versions"][0])
_MINISWE_VERSION = str(gen.AGENTS["mini-swe"]["versions"][0])
_MANIFEST = {"name": f"monet_code_{_MONET_VERSION}", "version": _MONET_VERSION,
             "repo": "https://github.com/YOU/monet_code", "ref": "aaa111", "token_env": "GH_TOKEN"}


def test_build_config_shape_and_source() -> None:
    cfg = gen.build_config("monet", "swe-bench-verified", _MANIFEST)
    assert cfg["run"]["name"] == f"eval-monet-{_MONET_VERSION}-swebench_verified"
    assert cfg["run"]["parallelism_eval_patches"] == 64   # two-phase eval fan-out (SWE-bench only)
    h = cfg["agent"]["harness"]
    assert h["name"] == "monet" and h["version"] == _MONET_VERSION
    assert h["source"] == {"repo": _MANIFEST["repo"], "ref": "aaa111", "token_env": "GH_TOKEN"}
    assert cfg["agent"]["extra_args"] == gen.AGENTS["monet"]["extra_args"]
    assert cfg["data"][0] == {"benchmark": "swe-bench-verified",
                              "dataset": "SWE-bench/SWE-bench_Verified", "split": "test"}
    # both knobs are emitted, so the artifact SHOWS them: the fallback timeout and the multiplier
    # that scales a task's declared budget (the annotations ride next to each value). The
    # multiplier is RUN-level — it applies to the first attempt, so it isn't a retry knob.
    assert cfg["run"]["retry"] == {"infra": 2}
    assert cfg["run"]["timeout_multiplier"] == 1.0


def test_infra_retry_lands_on_runconfig() -> None:
    # `retry` sits under the `run:` block, and the canonical loader lifts it onto RunConfig.retry.
    from beagle.cli._canonical import build_evaluation

    cfg = gen.build_config("opencode", "swe-bench-verified", _MANIFEST, smoke=True)
    assert cfg["run"]["retry"] == {"infra": 2}                    # smokes get it too
    assert cfg["run"]["timeout_multiplier"] == 1.0
    run_cfg, _ = build_evaluation(cfg)
    assert run_cfg.retry.infra == 2 and run_cfg.retry.content == 0


def test_build_config_public_manifest_omits_token() -> None:
    public = {k: v for k, v in _MANIFEST.items() if k != "token_env"}
    src = gen.build_config("mini-swe", "terminal_bench_2_1", public)["agent"]["harness"]["source"]
    assert "token_env" not in src and src["repo"] == public["repo"]


def test_deep_swe_parallelism_override() -> None:
    cfg = gen.build_config("opencode", "deep-swe", _MANIFEST)
    assert cfg["run"]["parallelism"] == 8            # benchmark override, not the default 32


def _seed_all(md: Path) -> None:
    md.mkdir(exist_ok=True)
    for _name, version, _label in gen.agent_cells():
        (md / f"{version}.json").write_text(json.dumps(
            {"version": version, "repo": f"https://github.com/YOU/{version}",
             "ref": "x" * 6, "token_env": "GH_TOKEN"}))


def test_generate_writes_the_whole_gate(tmp_path) -> None:
    """The gate must cover EVERY copy × benchmark — that is its job; examples/ is where the
    hand-written use cases live, and this script no longer writes there."""
    md, out = tmp_path / "m", tmp_path / "out"
    _seed_all(md)
    written, missing = gen.generate(manifest_dir=md, smoke_root=out, check=False)
    n_cells = len(list(gen.agent_cells())) * len(gen.BENCHMARKS) * len(gen.SMOKE_VARIANTS)
    assert written == n_cells and not missing
    for agent, _v, label in gen.agent_cells():                  # concrete source, no placeholder
        for bench in gen.BENCHMARKS:
            doc = yaml.safe_load((out / bench / f"{label}_smoke2.yaml").read_text())
            assert doc["agent"]["harness"]["name"] == agent
            assert doc["agent"]["harness"]["source"]["repo"].startswith("https://github.com/YOU/")


def test_build_config_smoke_variant() -> None:
    cfg = gen.build_config("mini-swe", "deep-swe", _MANIFEST, smoke=True)   # smoke=True == smoke2
    assert cfg["run"]["dir"] == "./tmp" and cfg["run"]["parallelism"] == 2
    assert cfg["run"]["name"] == f"miniswe-{_MINISWE_VERSION}-deepswe-smoke2"
    assert cfg["data"][0]["tasks"] == gen.smoke_tasks("deep-swe")   # the committed sample


def test_gate_is_one_config_per_combination(tmp_path) -> None:
    """One 2-task config per copy × benchmark — the gate answers 'does this combination
    work', and a second variant mirroring the sweep's knobs was the sweep's business."""
    md, smk = tmp_path / "m", tmp_path / "smk"
    _seed_all(md)
    w1, _ = gen.generate(manifest_dir=md, smoke_root=smk)
    n_cells = len(list(gen.agent_cells())) * len(gen.BENCHMARKS)
    assert w1 == n_cells * len(gen.SMOKE_VARIANTS)
    for _name, _version, label in gen.agent_cells():
        for bench, b in gen.BENCHMARKS.items():
            # Smokes are grouped by BENCHMARK and named after the copy, exactly like the eval tree
            # — one spelling of each benchmark, one naming rule for both variants.
            doc2 = yaml.safe_load((smk / bench / f"{label}_smoke2.yaml").read_text())
            assert doc2["run"]["parallelism"] == 2
            assert doc2["data"][0]["tasks"] == gen.smoke_tasks(bench)
            assert doc2["agent"]["harness"]["source"]["repo"].startswith("https://github.com/YOU/")


def test_generate_skips_unonboarded_agents(tmp_path) -> None:
    md = tmp_path / "m"
    md.mkdir()
    (md / "monet.json").write_text(json.dumps(
        {"version": _MONET_VERSION, "repo": "https://github.com/YOU/monet", "ref": "y" * 6}))
    out = tmp_path / "out"
    written, missing = gen.generate(manifest_dir=md, smoke_root=out, check=False)
    assert written == len(gen.BENCHMARKS) * len(gen.SMOKE_VARIANTS)   # only monet's row
    assert any("mini-swe" in m for m in missing) and any("opencode" in m for m in missing)


def test_manifests_by_version_raises_on_duplicate(tmp_path) -> None:
    (tmp_path / "a.json").write_text(json.dumps({"version": "1", "repo": "A", "ref": "x"}))
    (tmp_path / "b.json").write_text(json.dumps({"version": "1", "repo": "B", "ref": "y"}))
    with pytest.raises(SystemExit, match="share version"):
        gen._manifests_by_version(tmp_path)


def test_check_mode_writes_nothing_and_flags_missing(tmp_path) -> None:
    (tmp_path / "m.json").write_text(json.dumps(
        {"version": _MONET_VERSION, "repo": "R", "ref": "x", "token_env": "GH_TOKEN"}))  # monet only
    out = tmp_path / "out"
    rc = gen.main(["--manifest-dir", str(tmp_path), "--out", str(out), "--check"])
    assert rc == 1 and not out.exists()


def test_experiment_defaults_cover_the_whole_canonical_matrix() -> None:
    """The baseline sweep must default to EVERY onboarded agent × registered benchmark.

    These were hand-kept lists, so swe-rebench was silently missing from the one command that is
    supposed to regenerate everything — the configs existed only if you knew to pass --benches.
    """
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "experiments_gen", _P(__file__).resolve().parents[2] / "experiments/scripts/generate_eval_configs.py")
    assert spec and spec.loader
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)

    assert exp.DEF_BENCHES == list(exp.gen.BENCHMARKS)
    # one default per experiment COPY (label), not per adapter name
    assert exp.DEF_AGENTS == [label for _n, _v, label in exp.gen.agent_cells()]
    assert "swe-rebench" in exp.DEF_BENCHES


def test_artifact_names_are_symmetric_across_agents() -> None:
    # Every cell is named <harness>-<version>, whether or not the agent has a second copy. A rule
    # that keyed off sibling count would rename an agent's artifacts the moment someone added a
    # version to a DIFFERENT line of the same table.
    for name, version, label in gen.agent_cells():
        assert label == f"{name}-{version}"


def test_two_copies_of_one_harness_do_not_collide(monkeypatch, tmp_path) -> None:
    """harness.name and harness.version are separate fields precisely so baseline and candidate
    copies of the SAME harness can coexist. The generator has to carry that distinction into the
    artifact names, or copy B overwrites copy A's config and shares its run.name (colliding again
    in the results dir)."""
    monkeypatch.setitem(gen.AGENTS["monet"], "versions", ["20260826", "20260816"])

    cells = [(n, v, label) for n, v, label in gen.agent_cells() if n == "monet"]
    assert cells == [("monet", "20260826", "monet-20260826"),
                     ("monet", "20260816", "monet-20260816")]
    # ...and the OTHER agents' names are untouched by monet gaining a copy
    assert ("mini-swe", "v2.4.6", "mini-swe-v2.4.6") in list(gen.agent_cells())

    # the adapter is still `monet` in both; only the version (and the artifact name) differ
    cfgs = [gen.build_config("monet", "deep-swe", {"repo": "r", "ref": "x"}, version=v)
            for _n, v, _l in cells]
    assert [c["agent"]["harness"]["name"] for c in cfgs] == ["monet", "monet"]
    assert [c["agent"]["harness"]["version"] for c in cfgs] == ["20260826", "20260816"]
    assert cfgs[0]["run"]["name"] != cfgs[1]["run"]["name"]

    md, out = tmp_path / "m", tmp_path / "out"
    md.mkdir()
    for v in ("20260826", "20260816"):
        (md / f"{v}.json").write_text(json.dumps(
            {"version": v, "repo": f"https://github.com/YOU/monet_{v}", "ref": "z" * 6}))
    written, _missing = gen.generate(manifest_dir=md, smoke_root=out, check=False)
    assert written == 2 * len(gen.BENCHMARKS) * len(gen.SMOKE_VARIANTS)   # both copies generated
    for v in ("20260826", "20260816"):
        doc = yaml.safe_load((out / "deep-swe" / f"monet-{v}_smoke2.yaml").read_text())
        assert doc["agent"]["harness"] == {
            "name": "monet", "version": v,
            "source": {"repo": f"https://github.com/YOU/monet_{v}", "ref": "z" * 6}}


def test_two_copies_get_distinct_smoke_filenames(monkeypatch, tmp_path) -> None:
    # Copies share the benchmark dir, so the label (which carries the version) keeps them apart.
    monkeypatch.setitem(gen.AGENTS["monet"], "versions", ["20260826", "20260816"])
    md, smk = tmp_path / "m", tmp_path / "smk"
    md.mkdir()
    for v in ("20260826", "20260816"):
        (md / f"{v}.json").write_text(json.dumps({"version": v, "repo": "R", "ref": "z" * 6}))
    gen.generate(manifest_dir=md, smoke_root=smk)

    for v in ("20260826", "20260816"):
        assert (smk / "deep-swe" / f"monet-{v}_smoke2.yaml").exists()


def test_gate_is_grouped_by_benchmark(tmp_path) -> None:
    """`<benchmark>/<copy-label>_<variant>.yaml`. It used to be grouped by agent with each VARIANT
    naming its own file, so one directory held `terminal_bench_2_1_smoke2.yaml` beside
    `tb21_smoke1_gpt56sol.yaml` — two spellings of one benchmark."""
    md, smk = tmp_path / "m", tmp_path / "smk"
    _seed_all(md)
    gen.generate(manifest_dir=md, smoke_root=smk)

    assert {d.name for d in smk.iterdir()} == set(gen.BENCHMARKS)
    for _n, _v, label in gen.agent_cells():
        for variant in gen.SMOKE_VARIANTS:
            assert (smk / "deep-swe" / f"{label}_{variant}.yaml").exists()


def _tracked_docs() -> list[Path]:
    """Every tracked doc/script that could quote a generated name.

    Deliberately not a hardcoded list: the first version of these tests named five files and
    therefore missed two others that carried stale commands. The file list was the bug.
    """
    import subprocess

    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(["git", "ls-files", "*.md", "*.sh", "*.py"], cwd=root,
                         capture_output=True, text=True, check=True).stdout.split()
    skip = ("vendor/", "notes/", "tests/unit/")        # notes are historical; unit tests are these
    # `git ls-files` lists tracked paths, which may be deleted in the worktree mid-cleanup.
    return [p for f in out if not f.startswith(skip) if (p := root / f).exists()]


def test_documented_config_paths_are_ones_the_generator_writes() -> None:
    """Docs quote generated config paths, which carry the version — so they go stale exactly when
    an agent is re-onboarded. Fail here rather than in a user's terminal."""
    import re

    labels = {label for _n, _v, label in gen.agent_cells()}
    # both trees: examples/evaluation/<bench>/<label>.yaml and tests/smoke/<bench>/<label>_<variant>.yaml
    valid = set(labels) | {f"{lb}_{v}" for lb in labels for v in gen.SMOKE_VARIANTS}
    pattern = re.compile(r"(?:examples/evaluation|tests/smoke)/([\w.-]+)/([\w.-]+)\.yaml")
    checked = 0
    for path in _tracked_docs():
        for bench, stem in pattern.findall(path.read_text(encoding="utf-8")):
            assert bench in gen.BENCHMARKS, f"{path.name}: unknown benchmark dir {bench!r}"
            assert stem in valid, (
                f"{path.name}: references {bench}/{stem}.yaml, which the generator no longer "
                f"writes; current copies are {sorted(labels)}")
            checked += 1
    assert checked, "no documented config paths found — did the doc format change?"


def test_documented_baseline_stems_match_config_stem() -> None:
    """Same for the experiments sweep: its stems embed label + bench + model/effort/turns, so any
    of those changing rots the documented commands."""
    import re

    exp = _experiments_gen()
    pattern = re.compile(r"eval_baseline/([\w.-]+?)_([a-z0-9_]+)_([\w.-]+)_(\w+)_(\d+)")
    checked = 0
    for path in _tracked_docs():
        for label, short, model, effort, turns in pattern.findall(path.read_text(encoding="utf-8")):
            assert label in exp._CELLS, f"{path.name}: {label!r} is not an experiment copy"
            bench = next((b for b, cfg in gen.BENCHMARKS.items() if cfg["short"] == short), None)
            assert bench, f"{path.name}: {short!r} is not a benchmark short name"
            assert exp.config_stem(label, bench, model=model, effort=effort,
                                   max_turns=int(turns)) == f"{label}_{short}_{model}_{effort}_{turns}"
            checked += 1
    assert checked, "no documented baseline stems found — did the naming change?"


def _experiments_gen():
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "experiments_gen", root / "experiments/scripts/generate_eval_configs.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_documented_agent_selectors_are_accepted() -> None:
    """Every `--agents X` in a tracked doc or script must be a label the generator accepts; a bare
    harness name is rejected on purpose, and used to be what all of them passed."""
    import re

    exp = _experiments_gen()
    for path in _tracked_docs():
        text = path.read_text(encoding="utf-8")
        for match in re.findall(r'--agents ([\w.\- ]+?)(?:\s+--|\s*\\|\s*"|$)', text, re.MULTILINE):
            for token in match.split():
                assert token in exp._CELLS, (
                    f"{path.name}: --agents {token!r} is not a valid copy label "
                    f"(have {sorted(exp._CELLS)})")


def test_version_must_be_a_safe_path_component(monkeypatch) -> None:
    # A version reaches the config filename, the smoke filename and run.name (hence the results
    # dir), so a '/' would quietly create a nested directory instead of one artifact.
    monkeypatch.setitem(gen.AGENTS["monet"], "versions", ["release/1.2"])
    with pytest.raises(SystemExit, match="filename component"):
        list(gen.agent_cells())


def test_empty_versions_list_is_rejected(monkeypatch) -> None:
    # Silently generating zero configs for an agent is worse than refusing to start.
    monkeypatch.setitem(gen.AGENTS["monet"], "versions", [])
    with pytest.raises(SystemExit, match="lists no versions"):
        list(gen.agent_cells())


def test_timeout_knobs_are_annotated_in_the_generated_yaml() -> None:
    """A bare `timeout: 1800` at the bottom of a file reads as a per-run wall clock. The note has
    to travel WITH the value, and the multiplier — the knob people should actually reach for — has
    to be present at all, not omitted whenever it happens to be 1.0."""
    body = gen._dump("monet", "terminal_bench_2_1", gen.build_config(
        "monet", "terminal_bench_2_1", _MANIFEST))
    lines = body.splitlines()
    ti = next(i for i, ln in enumerate(lines) if ln.strip().startswith("timeout:"))
    mi = next(i for i, ln in enumerate(lines) if ln.strip().startswith("timeout_multiplier:"))
    assert "LAST RESORT" in lines[ti - 3]                      # note sits directly above the value
    assert "always wins" in lines[ti - 2]
    assert "knob for run length" in lines[mi - 2]
    assert lines[mi].strip() == "timeout_multiplier: 1.0"      # emitted even at the default


def test_version_with_a_trailing_newline_is_rejected(monkeypatch) -> None:
    # `$` also matches just before a final newline, so "1.2\n" passed validation and would have
    # produced a filename containing a newline. fullmatch + \Z is the fix.
    monkeypatch.setitem(gen.AGENTS["monet"], "versions", ["release-1.2\n"])
    with pytest.raises(SystemExit, match="filename component"):
        list(gen.agent_cells())


def test_no_tracked_file_puts_timeout_multiplier_under_retry() -> None:
    """It moved to run-level and RetryPolicy now REJECTS the old key, so any doc still showing it
    nested under `retry:` hands the reader a config that fails to load.

    Scans notes/ too — a live reference doc there (retry-coverage.md) kept the old shape precisely
    because the other doc tests skip that directory.
    """
    import re
    import subprocess

    root = Path(__file__).resolve().parents[2]
    tracked = subprocess.run(["git", "ls-files", "*.md", "*.yaml", "*.yml"], cwd=root,
                             capture_output=True, text=True, check=True).stdout.split()
    # `retry:` followed, within its indented block, by a timeout_multiplier line
    nested = re.compile(r"^(\s*)retry:\s*$(?:\n\1\s+.*$)*?\n\1\s+timeout_multiplier:", re.MULTILINE)
    offenders = [f for f in tracked
                 if not f.startswith("vendor/") and (root / f).exists()      # may be mid-deletion
                 and nested.search((root / f).read_text(encoding="utf-8"))]
    assert not offenders, (
        f"{offenders} nest timeout_multiplier under `retry:`; it is run-level "
        "(run.timeout_multiplier) and the old location is rejected at load")
