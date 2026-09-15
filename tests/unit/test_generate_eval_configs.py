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
    cfg = gen.build_config("monet", "swe-bench-verified", _MANIFEST, internal=True)
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


def test_public_profile_is_local_direct_and_excludes_monet() -> None:
    assert {name for name, _version, _label in gen.agent_cells()} == {"mini-swe", "opencode"}
    cfg = gen.build_config("opencode", "terminal_bench_2_1", _MANIFEST)

    assert cfg["run"]["runtime"] == "local"
    assert cfg["run"]["parallelism"] == 1
    assert cfg["agent"]["model"]["name"] == "gpt-5.6-sol"
    assert cfg["agent"]["effort"] == "medium"
    assert cfg["agent"]["provider"] == {"type": "direct", "name": "openai"}
    assert cfg["agent"]["forward_env"] == ["OPENAI_API_KEY"]
    with pytest.raises(ValueError, match="internal-only"):
        gen.build_config("monet", "terminal_bench_2_1", _MANIFEST)


def test_internal_profile_uses_gateway_cluster_and_includes_monet() -> None:
    assert "monet" in {name for name, _version, _label in gen.agent_cells(internal=True)}
    cfg = gen.build_config("monet", "terminal_bench_2_1", _MANIFEST, internal=True)

    assert cfg["run"]["runtime"] == "xrlenv-cluster"
    assert cfg["run"]["parallelism"] == 32
    assert cfg["agent"]["provider"] == {
        "type": "internal", "name": "llm-gateway-express-local-proxy"}
    assert cfg["agent"]["forward_env"] == gen._INTERNAL_FORWARD_ENV


def test_profile_defaults_can_be_overridden() -> None:
    cfg = gen.build_config(
        "opencode",
        "terminal_bench_2_1",
        _MANIFEST,
        runtime="xrlenv-cluster",
        provider={"type": "internal", "name": "custom-proxy"},
        forward_env=["CUSTOM_API_KEY"],
    )

    assert cfg["run"]["runtime"] == "xrlenv-cluster"
    assert cfg["agent"]["provider"] == {"type": "internal", "name": "custom-proxy"}
    assert cfg["agent"]["forward_env"] == ["CUSTOM_API_KEY"]

    internal_cfg = gen.build_config(
        "monet",
        "terminal_bench_2_1",
        _MANIFEST,
        internal=True,
        runtime="local",
        provider={"type": "internal", "name": "custom-proxy"},
        forward_env=["CUSTOM_API_KEY"],
    )
    assert internal_cfg["run"]["runtime"] == "local"
    assert internal_cfg["agent"]["provider"] == {"type": "internal", "name": "custom-proxy"}
    assert internal_cfg["agent"]["forward_env"] == ["CUSTOM_API_KEY"]


def test_provider_override_rejects_the_removed_scalar_form() -> None:
    with pytest.raises(ValueError, match="typed provider mapping"):
        gen.build_config(
            "opencode",
            "terminal_bench_2_1",
            _MANIFEST,
            provider="custom-proxy",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="invalid provider configuration"):
        gen.build_config(
            "opencode",
            "terminal_bench_2_1",
            _MANIFEST,
            provider={"type": "internal"},
        )


def test_provider_cli_override_builds_type_specific_fields() -> None:
    assert gen.provider_override(
        provider_type="direct", name="anthropic"
    ) == {"type": "direct", "name": "anthropic"}
    assert gen.provider_override(
        provider_type="internal", name="deployment-gateway"
    ) == {"type": "internal", "name": "deployment-gateway"}
    assert gen.provider_override(
        provider_type="gateway",
        name="org",
        api_base="https://gw.example/v1",
        api_key_env="ORG_KEY",
        auth_header="X-Api-Key",
    ) == {
        "type": "gateway",
        "name": "org",
        "extra_args": {
            "api_base": "https://gw.example/v1",
            "api_key_env": "ORG_KEY",
            "auth_header": "X-Api-Key",
        },
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"provider_type": "direct"}, "provider-name"),
        ({"provider_type": "internal"}, "provider-name"),
        ({"provider_type": "gateway", "api_base": "https://gw/v1"}, "provider-name"),
        ({"provider_type": "gateway", "name": "org"}, "provider-api-base"),
        ({"provider_type": "gateway", "name": "org", "api_base": "https://gw/v1",
          "auth_header": "X-Api-Key"}, "provider-api-key-env"),
        ({"provider_type": "direct", "name": "openai", "api_base": "https://gw/v1"},
         "gateway fields"),
        ({"provider_type": None, "name": "openai"}, "require --provider-type"),
    ],
)
def test_provider_cli_override_rejects_invalid_field_combinations(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        gen.provider_override(**kwargs)


def test_deep_swe_parallelism_override() -> None:
    cfg = gen.build_config("opencode", "deep-swe", _MANIFEST, internal=True)
    assert cfg["run"]["parallelism"] == 8            # benchmark override, not the default 32


def _seed_all(md: Path, *, internal: bool = False) -> None:
    md.mkdir(exist_ok=True)
    for _name, version, _label in gen.agent_cells(internal=internal):
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


def test_smoke_requires_seeded_tasks(monkeypatch) -> None:
    monkeypatch.setattr(gen, "smoke_tasks", lambda _bench: [])

    with pytest.raises(SystemExit, match="--reseed-smoke-tasks"):
        gen.build_config("mini-swe", "deep-swe", _MANIFEST, smoke=True)


def test_internal_smoke_does_not_overprovision_patch_evaluators() -> None:
    cfg = gen.build_config(
        "mini-swe", "swe-bench-verified", _MANIFEST, smoke=True, internal=True,
    )

    assert cfg["run"]["parallelism"] == 2
    assert cfg["run"]["parallelism_eval_patches"] == 2


def test_gate_is_one_config_per_combination(tmp_path) -> None:
    """One 2-task config per copy × benchmark — the gate answers 'does this combination
    work', and a second variant mirroring the sweep's knobs was the sweep's business."""
    md, smk = tmp_path / "m", tmp_path / "smk"
    _seed_all(md)
    w1, _ = gen.generate(manifest_dir=md, smoke_root=smk)
    n_cells = len(list(gen.agent_cells())) * len(gen.BENCHMARKS)
    assert w1 == n_cells * len(gen.SMOKE_VARIANTS)
    for _name, _version, label in gen.agent_cells():
        for bench in gen.BENCHMARKS:
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
    written, missing = gen.generate(manifest_dir=md, smoke_root=out, check=False, internal=True)
    assert written == len(gen.BENCHMARKS) * len(gen.SMOKE_VARIANTS)   # only monet's row
    assert any("mini-swe" in m for m in missing) and any("opencode" in m for m in missing)
    assert f"mini-swe-{_MINISWE_VERSION} ({_MINISWE_VERSION})" in missing
    assert all("(vv" not in item for item in missing)


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


def test_experiment_generator_profiles(tmp_path) -> None:
    exp = _experiments_gen()
    manifests = tmp_path / "manifests"
    _seed_all(manifests, internal=True)

    public_out = tmp_path / "public"
    assert exp.main([
        "--manifest-dir", str(manifests), "--out", str(public_out),
        "--results", str(tmp_path / "results"),
    ]) == 0
    public_files = sorted(public_out.glob("*.yaml"))
    assert public_files
    assert not any(path.name.startswith("monet-") for path in public_files)
    public_cfg = yaml.safe_load(public_files[0].read_text(encoding="utf-8"))
    assert public_cfg["run"]["runtime"] == "local"
    assert public_cfg["run"]["parallelism"] == 1
    assert public_cfg["agent"]["provider"] == {"type": "direct", "name": "openai"}
    assert public_cfg["agent"]["forward_env"] == ["OPENAI_API_KEY"]
    public_swe = yaml.safe_load(next(
        public_out.glob("*_swebench_verified_*.yaml")
    ).read_text(encoding="utf-8"))
    assert public_swe["run"]["parallelism_eval_patches"] == 1

    internal_out = tmp_path / "internal"
    assert exp.main([
        "--internal", "--manifest-dir", str(manifests), "--out", str(internal_out),
        "--results", str(tmp_path / "results"),
    ]) == 0
    monet_file = next(internal_out.glob("monet-*_swebench_verified_*.yaml"))
    internal_cfg = yaml.safe_load(monet_file.read_text(encoding="utf-8"))
    assert internal_cfg["run"]["runtime"] == "xrlenv-cluster"
    assert internal_cfg["run"]["parallelism_eval_patches"] == 64
    assert internal_cfg["agent"]["provider"] == {
        "type": "internal", "name": "llm-gateway-express-local-proxy"}
    assert internal_cfg["agent"]["forward_env"] == gen._INTERNAL_FORWARD_ENV


def test_experiment_generator_rejects_monet_without_internal(tmp_path) -> None:
    exp = _experiments_gen()
    with pytest.raises(SystemExit, match="2"):
        exp.main([
            "--agents", f"monet-{_MONET_VERSION}",
            "--manifest-dir", str(tmp_path),
            "--out", str(tmp_path / "out"),
        ])


def test_experiment_missing_hint_does_not_double_version_prefix(
    tmp_path, capsys: pytest.CaptureFixture[str],
) -> None:
    exp = _experiments_gen()

    assert exp.main([
        "--check",
        "--manifest-dir", str(tmp_path),
        "--out", str(tmp_path / "out"),
    ]) == 1
    output = capsys.readouterr().out
    assert f"mini-swe-{_MINISWE_VERSION} ({_MINISWE_VERSION})" in output
    assert "(vv" not in output


def test_experiment_generator_profile_overrides(tmp_path) -> None:
    exp = _experiments_gen()
    manifests = tmp_path / "manifests"
    _seed_all(manifests)
    out = tmp_path / "out"

    assert exp.main([
        "--agents", f"opencode-{gen.AGENTS['opencode']['versions'][0]}",
        "--benches", "terminal_bench_2_1",
        "--runtime", "xrlenv-cluster",
        "--provider-type", "gateway",
        "--provider-name", "custom-proxy",
        "--provider-api-base", "https://gw.example/v1",
        "--provider-api-key-env", "CUSTOM_API_KEY",
        "--provider-auth-header", "X-Api-Key",
        "--forward-env", "CUSTOM_API_KEY",
        "--parallelism", "7",
        "--manifest-dir", str(manifests),
        "--out", str(out),
    ]) == 0
    cfg = yaml.safe_load(next(out.glob("*.yaml")).read_text(encoding="utf-8"))
    assert cfg["run"]["runtime"] == "xrlenv-cluster"
    assert cfg["run"]["parallelism"] == 7
    assert cfg["agent"]["provider"] == {
        "type": "gateway",
        "name": "custom-proxy",
        "extra_args": {
            "api_base": "https://gw.example/v1",
            "api_key_env": "CUSTOM_API_KEY",
            "auth_header": "X-Api-Key",
        },
    }
    assert cfg["agent"]["forward_env"] == ["CUSTOM_API_KEY"]


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

    cells = [(n, v, label) for n, v, label in gen.agent_cells(internal=True) if n == "monet"]
    assert cells == [("monet", "20260826", "monet-20260826"),
                     ("monet", "20260816", "monet-20260816")]
    # ...and the OTHER agents' names are untouched by monet gaining a copy
    assert ("mini-swe", "v2.4.6", "mini-swe-v2.4.6") in list(gen.agent_cells())

    # the adapter is still `monet` in both; only the version (and the artifact name) differ
    cfgs = [gen.build_config("monet", "deep-swe", {"repo": "r", "ref": "x"}, version=v,
                             internal=True)
            for _n, v, _l in cells]
    assert [c["agent"]["harness"]["name"] for c in cfgs] == ["monet", "monet"]
    assert [c["agent"]["harness"]["version"] for c in cfgs] == ["20260826", "20260816"]
    assert cfgs[0]["run"]["name"] != cfgs[1]["run"]["name"]

    md, out = tmp_path / "m", tmp_path / "out"
    md.mkdir()
    for v in ("20260826", "20260816"):
        (md / f"{v}.json").write_text(json.dumps(
            {"version": v, "repo": f"https://github.com/YOU/monet_{v}", "ref": "z" * 6}))
    written, _missing = gen.generate(manifest_dir=md, smoke_root=out, check=False, internal=True)
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
    gen.generate(manifest_dir=md, smoke_root=smk, internal=True)

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


def test_documented_config_paths_are_ones_the_generator_writes(tmp_path) -> None:
    """Docs quote generated config paths, which carry the version — so they go stale exactly when
    an agent is re-onboarded. Fail here rather than in a user's terminal."""
    import re

    labels = {label for _n, _v, label in gen.agent_cells()}
    # Evolution and evaluation share tests/smoke/, but have separate generators.
    # Generate both into a temporary tree so a stale evolution suffix still fails.
    spec = importlib.util.spec_from_file_location(
        "documented_evolution_gen", _PATH.with_name("generate_evolution_config.py")
    )
    assert spec and spec.loader
    evolution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evolution)
    manifests, out = tmp_path / "manifests", tmp_path / "smoke"
    _seed_all(manifests)
    for path in manifests.glob("*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["dir"] = "../beagle-experiments/example"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    _, missing_eval = gen.generate(manifest_dir=manifests, smoke_root=out)
    _, missing_evolution = evolution.generate(manifest_dir=manifests, out_root=out)
    assert not missing_eval and not missing_evolution
    written = {path.relative_to(out).as_posix() for path in out.rglob("*.yaml")}
    pattern = re.compile(r"(examples/evaluation|tests/smoke)/([\w.-]+)/([\w.-]+)\.yaml")
    checked = 0
    for path in _tracked_docs():
        for tree, bench, stem in pattern.findall(path.read_text(encoding="utf-8")):
            assert bench in gen.BENCHMARKS, f"{path.name}: unknown benchmark dir {bench!r}"
            supported = (f"{bench}/{stem}.yaml" in written if tree == "tests/smoke"
                         else stem in labels)
            assert supported, (
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
        list(gen.agent_cells(internal=True))


def test_empty_versions_list_is_rejected(monkeypatch) -> None:
    # Silently generating zero configs for an agent is worse than refusing to start.
    monkeypatch.setitem(gen.AGENTS["monet"], "versions", [])
    with pytest.raises(SystemExit, match="lists no versions"):
        list(gen.agent_cells(internal=True))


def test_timeout_knobs_are_annotated_in_the_generated_yaml() -> None:
    """A bare `timeout: 1800` at the bottom of a file reads as a per-run wall clock. The note has
    to travel WITH the value, and the multiplier — the knob people should actually reach for — has
    to be present at all, not omitted whenever it happens to be 1.0."""
    body = gen._dump("monet", "terminal_bench_2_1", gen.build_config(
        "monet", "terminal_bench_2_1", _MANIFEST, internal=True))
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
        list(gen.agent_cells(internal=True))


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
