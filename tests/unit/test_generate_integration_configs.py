"""Provider-routing live integration config generation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from beagle.cli._canonical import build_evaluation

_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "integration"
    / "generate_provider_matrix.py"
)
_SPEC = importlib.util.spec_from_file_location("generate_provider_matrix", _PATH)
matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(matrix)  # type: ignore[union-attr]


def _seed_manifests(root: Path) -> None:
    root.mkdir()
    for harness, agent in matrix.AGENTS.items():
        version = agent["version"]
        (root / f"{harness}.json").write_text(
            json.dumps({
                "version": version,
                "repo": f"https://github.com/example/{harness}",
                "ref": "abc123",
                "token_env": "GH_TOKEN",
            }),
            encoding="utf-8",
        )


def test_generates_every_available_agent_route_pair(tmp_path) -> None:
    manifests = tmp_path / "manifests"
    out = tmp_path / "generated"
    _seed_manifests(manifests)

    paths = matrix.generate(manifest_dir=manifests, out=out)

    expected = {
        (agent, route)
        for agent in matrix.AGENTS
        for route in matrix.ROUTES
        if (agent, route) not in matrix.UNSUPPORTED_PAIRS
    }
    assert len(paths) == len(expected)
    for path in paths:
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
        assert raw["run"]["parallelism"] == 1
        assert raw["run"]["timeout_multiplier"] == 0.1
        assert raw["run"]["debug_max_agent_wall_time_sec"] == 360
        assert raw["agent"]["max_turns"] == 3
        assert raw["data"] == [{
            "benchmark": "deep-swe",
            "tasks": ["httpx-multipart-response-parsing"],
        }]
        build_evaluation(raw)
        if path.name.startswith("opencode__"):
            assert "no max-turn support" not in text

    assert not (out / "monet__gateway__deep-swe.yaml").exists()


def test_debug_wall_time_override_loads_canonically(tmp_path) -> None:
    manifests = tmp_path / "manifests"
    _seed_manifests(manifests)
    paths = matrix.generate(
        manifest_dir=manifests,
        out=tmp_path / "generated",
        agents=["opencode"],
        routes=["direct-api"],
        debug_max_agent_wall_time_sec=600,
    )
    raw = yaml.safe_load(paths[0].read_text(encoding="utf-8"))
    assert raw["run"]["debug_max_agent_wall_time_sec"] == 600
    cfg, _ = build_evaluation(raw)
    assert cfg.debug_max_agent_wall_time_sec == 600


def test_route_specific_provider_fields_and_credentials() -> None:
    direct = matrix.ROUTES["direct-api"]
    assert direct == {
        "provider": {"type": "direct", "name": "openai"},
        "forward_env": ["OPENAI_API_KEY"],
    }

    if "internal" in matrix.ROUTES:
        internal = matrix.ROUTES["internal"]
        assert internal["provider"] == {
            "type": "internal",
            "name": "llm-gateway-express-local-proxy",
        }
        assert "LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL" in internal["forward_env"]

    if "gateway" in matrix.ROUTES:
        gateway = matrix.ROUTES["gateway"]
        assert gateway["provider"] == {
            "type": "gateway",
            "name": "SFR-GATEWAY",
            "extra_args": {
                "api_base": "https://gateway.salesforceresearch.ai/openai/process/v1/",
                "api_key_env": "SFR_GATEWAY_API_KEY",
                "auth_header": "X-Api-Key",
            },
        }
        assert gateway["forward_env"] == []


def test_generated_yaml_contains_no_secret_values(tmp_path) -> None:
    manifests = tmp_path / "manifests"
    _seed_manifests(manifests)
    paths = matrix.generate(manifest_dir=manifests, out=tmp_path / "generated")

    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    if "gateway" in matrix.ROUTES:
        assert "SFR_GATEWAY_API_KEY" in combined
    assert "sk-" not in combined
