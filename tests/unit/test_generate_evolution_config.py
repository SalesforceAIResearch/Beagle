"""Tests for the matrix-backed DarwinX evolution-smoke generator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_evolution_config.py"
_SPEC = importlib.util.spec_from_file_location("generate_evolution_config", _PATH)
gen = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(gen)

OPENCODE_VERSION = str(gen.gen.AGENTS["opencode"]["versions"][0])
MINISWE_VERSION = str(gen.gen.AGENTS["mini-swe"]["versions"][0])
MONET_VERSION = str(gen.gen.AGENTS["monet"]["versions"][0])


def _manifest(version: str, name: str) -> dict:
    return {
        "profile": name,
        "version": version,
        "repo": f"https://github.com/example/{name}",
        "ref": "a" * 40,
        "token_env": "GH_TOKEN",
        "upstream": f"https://github.com/upstream/{name}",
        "dir": f"../beagle-experiments/{name}",
    }


def _seed_manifests(root: Path, *, internal: bool = False) -> None:
    root.mkdir(parents=True)
    for agent, version, label in gen.gen.agent_cells(internal=internal):
        (root / f"{label}.json").write_text(
            json.dumps(_manifest(version, agent)), encoding="utf-8"
        )


def test_public_opencode_smoke_uses_matrix_profile_and_two_seeded_tasks() -> None:
    manifest = _manifest(OPENCODE_VERSION, "opencode")
    raw = gen.build_config(
        "opencode",
        "terminal_bench_2_1",
        manifest,
        version=OPENCODE_VERSION,
    )

    assert raw["run"] == {
        "dir": "./tmp",
        "name": f"opencode-{OPENCODE_VERSION}-tb21-{gen.SUFFIX}",
        "runtime": "local",
        "parallelism": 2,
    }
    assert raw["evolvee"]["harness"]["source"] == {
        "repo": manifest["repo"],
        "ref": manifest["ref"],
        "token_env": "GH_TOKEN",
        "dir": manifest["dir"],
        "upstream": manifest["upstream"],
    }
    assert raw["evolvee"]["provider"] == {"type": "direct", "name": "openai"}
    assert raw["evolver"] == {
        "harness": {"name": "cursor-agent"},
        "model": {"name": "auto"},
    }
    assert raw["data"][0]["tasks"] == gen.gen.smoke_tasks("terminal_bench_2_1")
    assert len(raw["data"][0]["tasks"]) == 2
    assert raw["algorithm"]["hparams"] == gen.SMOKE_HPARAMS
    gen.validate_config(raw)


def test_generator_is_not_hardcoded_to_opencode() -> None:
    manifest = _manifest(MINISWE_VERSION, "mini-swe")
    raw = gen.build_config(
        "mini-swe",
        "terminal_bench_2_1",
        manifest,
        version=MINISWE_VERSION,
    )

    assert raw["evolvee"]["harness"]["name"] == "mini-swe"
    assert raw["evolvee"]["extra_args"] == gen.gen.AGENTS["mini-swe"]["extra_args"]
    gen.validate_config(raw)


def test_internal_profile_includes_monet_and_internal_routing() -> None:
    manifest = _manifest(MONET_VERSION, "monet")
    raw = gen.build_config(
        "monet",
        "terminal_bench_2_1",
        manifest,
        version=MONET_VERSION,
        internal=True,
    )

    assert raw["run"]["runtime"] == "xrlenv-cluster"
    assert raw["evolvee"]["provider"]["type"] == "internal"
    assert raw["evolver"]["model"]["name"] == "gpt-5.5-high"
    gen.validate_config(raw)


def test_generate_writes_one_smoke_per_public_copy_and_benchmark(tmp_path) -> None:
    manifests = tmp_path / "manifests"
    out = tmp_path / "smoke"
    _seed_manifests(manifests)

    written, missing = gen.generate(manifest_dir=manifests, out_root=out)

    expected = len(list(gen.gen.agent_cells())) * len(gen.gen.BENCHMARKS)
    assert written == expected and not missing
    for agent, version, label in gen.gen.agent_cells():
        for benchmark in gen.gen.BENCHMARKS:
            path = out / benchmark / f"{label}_{gen.SUFFIX}.yaml"
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert raw["evolvee"]["harness"]["name"] == agent
            assert raw["evolvee"]["harness"]["version"] == version
            assert raw["evolvee"]["harness"]["source"]["repo"].startswith(
                "https://github.com/example/"
            )


def test_generate_can_select_one_copy_and_benchmark(tmp_path) -> None:
    manifests = tmp_path / "manifests"
    out = tmp_path / "smoke"
    _seed_manifests(manifests)
    label = gen.gen._label("opencode", OPENCODE_VERSION)

    written, missing = gen.generate(
        manifest_dir=manifests,
        out_root=out,
        labels=[label],
        benchmarks=["terminal_bench_2_1"],
    )

    assert written == 1 and not missing
    assert (out / "terminal_bench_2_1" / f"{label}_{gen.SUFFIX}.yaml").exists()


def test_check_lists_missing_without_writing(tmp_path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    out = tmp_path / "smoke"

    written, missing = gen.generate(
        manifest_dir=manifests,
        out_root=out,
        check=True,
    )

    assert written == 0
    assert missing
    assert not out.exists()


def test_missing_checkout_path_fails_clearly() -> None:
    manifest = _manifest(OPENCODE_VERSION, "opencode")
    del manifest["dir"]

    with pytest.raises(ValueError, match="no 'dir'"):
        gen.build_config(
            "opencode",
            "terminal_bench_2_1",
            manifest,
            version=OPENCODE_VERSION,
        )


def test_unknown_selection_fails_clearly(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown agent copy"):
        gen.generate(
            manifest_dir=tmp_path,
            labels=["not-a-copy"],
            check=True,
        )
    with pytest.raises(ValueError, match="unknown benchmark"):
        gen.generate(
            manifest_dir=tmp_path,
            benchmarks=["not-a-benchmark"],
            check=True,
        )
