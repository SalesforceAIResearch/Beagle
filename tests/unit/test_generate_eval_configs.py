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

_MANIFEST = {"name": "monet_code_20260816", "version": "20260816",
             "repo": "https://github.com/YOU/monet_code", "ref": "aaa111", "token_env": "GH_TOKEN"}


def test_build_config_shape_and_source() -> None:
    cfg = gen.build_config("monet", "swe-bench-verified", _MANIFEST)
    assert cfg["run"]["name"] == "eval-monet-swebench_verified"
    assert cfg["run"]["parallelism_eval_patches"] == 64   # two-phase eval fan-out (SWE-bench only)
    h = cfg["agent"]["harness"]
    assert h["name"] == "monet" and h["version"] == "20260816"
    assert h["source"] == {"repo": _MANIFEST["repo"], "ref": "aaa111", "token_env": "GH_TOKEN"}
    assert cfg["agent"]["extra_args"] == gen.AGENTS["monet"]["extra_args"]
    assert cfg["data"][0] == {"benchmark": "swe-bench-verified",
                              "dataset": "SWE-bench/SWE-bench_Verified", "split": "test"}
    assert cfg["run"]["retry"] == {"infra": 2}       # infra-transient retry on by default


def test_infra_retry_lands_on_runconfig() -> None:
    # `retry` sits under the `run:` block, and the canonical loader lifts it onto RunConfig.retry.
    from beagle.cli._canonical import build_evaluation

    cfg = gen.build_config("opencode", "swe-bench-verified", _MANIFEST, smoke=True)
    assert cfg["run"]["retry"] == {"infra": 2}        # smokes get it too
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
    for a in gen.AGENTS.values():
        (md / f"{a['version']}.json").write_text(json.dumps(
            {"version": a["version"], "repo": f"https://github.com/YOU/{a['version']}",
             "ref": "x" * 6, "token_env": "GH_TOKEN"}))


def test_generate_writes_full_matrix(tmp_path) -> None:
    md, out = tmp_path / "m", tmp_path / "out"
    _seed_all(md)
    written, missing = gen.generate(manifest_dir=md, out_root=out, check=False)  # default: no smokes
    assert written == len(gen.AGENTS) * len(gen.BENCHMARKS) and not missing
    for agent in gen.AGENTS:                                    # concrete source, no placeholder
        for bench in gen.BENCHMARKS:
            doc = yaml.safe_load((out / bench / f"{agent}.yaml").read_text())
            assert doc["agent"]["harness"]["name"] == agent
            assert doc["agent"]["harness"]["source"]["repo"].startswith("https://github.com/YOU/")


def test_build_config_smoke_variant() -> None:
    cfg = gen.build_config("mini-swe", "deep-swe", _MANIFEST, smoke=True)   # smoke=True == smoke2
    assert cfg["run"]["dir"] == "./tmp" and cfg["run"]["parallelism"] == 2
    assert cfg["run"]["name"] == "miniswe-deepswe-smoke2"       # hyphen dropped in the name prefix
    assert cfg["data"][0]["tasks"] == gen.BENCHMARKS["deep-swe"]["smoke_tasks"]


def test_build_config_smoke1_gpt56sol_variant() -> None:
    # The gpt-5.6-sol smoke1 mirrors the eval_baseline sweep knobs on a SINGLE task, so a pre-sweep
    # smoke exercises the same shape. It's a --smoke output now (was a hand-maintained file that drifted).
    cfg = gen.build_config("monet", "terminal_bench_2_1", _MANIFEST,
                           variant=gen.SMOKE_VARIANTS["smoke1-gpt56sol"])
    assert cfg["run"]["name"] == "monet-tb21-smoke-gpt56sol" and cfg["run"]["dir"] == "./tmp"
    assert cfg["run"]["parallelism"] == 1
    assert cfg["agent"]["model"]["name"] == "gpt-5.6-sol"
    assert cfg["agent"]["effort"] == "medium" and cfg["agent"]["max_turns"] == 200
    assert cfg["data"][0]["tasks"] == ["bn-fit-modify"]        # first of the curated smoke_tasks


def test_smoke_flag_emits_only_smokes(tmp_path) -> None:
    md, out, smk = tmp_path / "m", tmp_path / "out", tmp_path / "smk"
    _seed_all(md)
    # default (no --smoke): eval only, nothing under tests/smoke
    w0, _ = gen.generate(manifest_dir=md, out_root=out, smoke_root=smk, smoke=False)
    assert w0 == len(gen.AGENTS) * len(gen.BENCHMARKS) and not smk.exists()
    # --smoke: smokes ONLY (every agent × benchmark × variant); no eval configs written this pass
    out2 = tmp_path / "out2"
    w1, _ = gen.generate(manifest_dir=md, out_root=out2, smoke_root=smk, smoke=True)
    n_cells = sum(1 for a in gen.AGENTS.values() if a.get("smoke_dir")) * len(gen.BENCHMARKS)
    assert w1 == n_cells * len(gen.SMOKE_VARIANTS) and not out2.exists()
    for a in gen.AGENTS.values():
        for bench, b in gen.BENCHMARKS.items():
            # smoke2 variant: the original 2-task check on the defaults
            doc2 = yaml.safe_load((smk / a["smoke_dir"] / b["smoke_file"]).read_text())
            assert doc2["run"]["parallelism"] == 2 and doc2["data"][0]["tasks"] == b["smoke_tasks"]
            assert doc2["agent"]["harness"]["source"]["repo"].startswith("https://github.com/YOU/")
            # smoke1-gpt56sol variant: single task, gpt-5.6-sol sweep knobs, synced ref
            doc1 = yaml.safe_load((smk / a["smoke_dir"] / f"{b['short']}_smoke1_gpt56sol.yaml").read_text())
            assert doc1["run"]["parallelism"] == 1 and doc1["data"][0]["tasks"] == b["smoke_tasks"][:1]
            assert doc1["agent"]["model"]["name"] == "gpt-5.6-sol" and doc1["agent"]["max_turns"] == 200


def test_generate_skips_unonboarded_agents(tmp_path) -> None:
    md = tmp_path / "m"
    md.mkdir()
    (md / "monet.json").write_text(json.dumps(
        {"version": "20260816", "repo": "https://github.com/YOU/monet", "ref": "y" * 6}))
    out = tmp_path / "out"
    written, missing = gen.generate(manifest_dir=md, out_root=out, check=False)
    assert written == len(gen.BENCHMARKS)                       # only monet's row
    assert any("mini-swe" in m for m in missing) and any("opencode" in m for m in missing)


def test_manifests_by_version_raises_on_duplicate(tmp_path) -> None:
    (tmp_path / "a.json").write_text(json.dumps({"version": "1", "repo": "A", "ref": "x"}))
    (tmp_path / "b.json").write_text(json.dumps({"version": "1", "repo": "B", "ref": "y"}))
    with pytest.raises(SystemExit, match="share version"):
        gen._manifests_by_version(tmp_path)


def test_check_mode_writes_nothing_and_flags_missing(tmp_path) -> None:
    (tmp_path / "m.json").write_text(json.dumps(
        {"version": "20260816", "repo": "R", "ref": "x", "token_env": "GH_TOKEN"}))   # monet only
    out = tmp_path / "out"
    rc = gen.main(["--manifest-dir", str(tmp_path), "--out", str(out), "--check"])
    assert rc == 1 and not out.exists()
