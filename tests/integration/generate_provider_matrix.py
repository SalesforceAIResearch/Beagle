#!/usr/bin/env python3
"""Generate a small live provider-routing matrix for DeepSWE.

The generated YAML files are operator-local (sources come from ``.beagle/agents``) and
land under ``tests/integration/generated/``. Each config runs one task at parallelism one.
Run them with ``beagle evaluate --config <path>``; these are live tests and spend tokens.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_GENERATOR = REPO_ROOT / "scripts" / "generate_eval_configs.py"
DEFAULT_MANIFEST_DIR = REPO_ROOT / ".beagle" / "agents"
DEFAULT_OUT = REPO_ROOT / "tests" / "integration" / "generated"
DEFAULT_TASK = "httpx-multipart-response-parsing"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_DEBUG_MAX_AGENT_WALL_TIME_SEC = 360.0


def _load_canonical():
    spec = importlib.util.spec_from_file_location(
        "beagle_canonical_eval_generator", CANONICAL_GENERATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {CANONICAL_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CANONICAL = _load_canonical()

AGENTS = {
    "opencode": {
        "version": str(_CANONICAL.AGENTS["opencode"]["versions"][0]),
        "extra_args": _CANONICAL.AGENTS["opencode"]["extra_args"],
    },
}

ROUTES = {
    "direct-api": {
        "provider": {"type": "direct", "name": "openai"},
        "forward_env": ["OPENAI_API_KEY"],
    },
}

UNSUPPORTED_PAIRS = {
}


def _manifests_by_version(manifest_dir: Path) -> dict[str, dict]:
    manifests: dict[str, dict] = {}
    for path in sorted(manifest_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        version = str(data.get("version") or "")
        if version:
            manifests[version] = data
    return manifests


def build_config(
    harness: str,
    route_name: str,
    manifest: dict,
    *,
    task: str = DEFAULT_TASK,
    model: str = DEFAULT_MODEL,
    max_turns: int = 3,
    timeout_multiplier: float = 0.1,
    debug_max_agent_wall_time_sec: float | None = DEFAULT_DEBUG_MAX_AGENT_WALL_TIME_SEC,
    runtime: str = "xrlenv-cluster",
) -> dict:
    """Build one harness × route live-test config."""
    agent = AGENTS[harness]
    route = ROUTES[route_name]
    source = {"repo": manifest["repo"], "ref": manifest["ref"]}
    if manifest.get("token_env"):
        source["token_env"] = manifest["token_env"]
    run = {
        "dir": "./tmp/provider-matrix",
        "name": f"integration-{harness}-{route_name}-deep-swe",
        "runtime": runtime,
        "parallelism": 1,
        "timeout_multiplier": timeout_multiplier,
    }
    if debug_max_agent_wall_time_sec is not None:
        run["debug_max_agent_wall_time_sec"] = debug_max_agent_wall_time_sec
    return {
        "run": run,
        "agent": {
            "harness": {
                "name": harness,
                "version": agent["version"],
                "source": source,
            },
            "model": {"name": model},
            "provider": route["provider"],
            "forward_env": list(route["forward_env"]),
            "effort": "medium",
            "max_turns": max_turns,
            "timeout": 1800,
            "extra_args": agent["extra_args"],
        },
        "data": [{"benchmark": "deep-swe", "tasks": [task]}],
    }


def generate(
    *,
    manifest_dir: Path = DEFAULT_MANIFEST_DIR,
    out: Path = DEFAULT_OUT,
    agents: list[str] | None = None,
    routes: list[str] | None = None,
    task: str = DEFAULT_TASK,
    model: str = DEFAULT_MODEL,
    max_turns: int = 3,
    timeout_multiplier: float = 0.1,
    debug_max_agent_wall_time_sec: float | None = DEFAULT_DEBUG_MAX_AGENT_WALL_TIME_SEC,
    runtime: str = "xrlenv-cluster",
    clean: bool = False,
) -> list[Path]:
    """Write the selected matrix and return generated paths."""
    selected_agents = agents or list(AGENTS)
    selected_routes = routes or list(ROUTES)
    manifests = _manifests_by_version(manifest_dir)
    if clean and out.exists():
        for path in out.glob("*__*__deep-swe.yaml"):
            path.unlink()
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for harness in selected_agents:
        version = AGENTS[harness]["version"]
        manifest = manifests.get(version)
        if manifest is None:
            raise SystemExit(
                f"no onboarded manifest with version {version!r} for {harness}; "
                f"expected one under {manifest_dir}"
            )
        for route_name in selected_routes:
            if (harness, route_name) in UNSUPPORTED_PAIRS:
                continue
            config = build_config(
                harness,
                route_name,
                manifest,
                task=task,
                model=model,
                max_turns=max_turns,
                timeout_multiplier=timeout_multiplier,
                debug_max_agent_wall_time_sec=debug_max_agent_wall_time_sec,
                runtime=runtime,
            )
            path = out / f"{harness}__{route_name}__deep-swe.yaml"
            header = (
                "# GENERATED by tests/integration/generate_provider_matrix.py; "
                "live test, spends tokens.\n"
            )
            path.write_text(
                header + yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--agents", nargs="+", choices=list(AGENTS))
    parser.add_argument("--routes", nargs="+", choices=list(ROUTES))
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--timeout-multiplier", type=float, default=0.1)
    parser.add_argument(
        "--debug-max-agent-wall-time-sec",
        type=float,
        default=DEFAULT_DEBUG_MAX_AGENT_WALL_TIME_SEC,
    )
    parser.add_argument("--runtime", choices=["local", "xrlenv-cluster"],
                        default="xrlenv-cluster")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args(argv)

    paths = generate(
        manifest_dir=args.manifest_dir,
        out=args.out,
        agents=args.agents,
        routes=args.routes,
        task=args.task,
        model=args.model,
        max_turns=args.max_turns,
        timeout_multiplier=args.timeout_multiplier,
        debug_max_agent_wall_time_sec=args.debug_max_agent_wall_time_sec,
        runtime=args.runtime,
        clean=args.clean,
    )
    for path in paths:
        print(path.relative_to(REPO_ROOT))
    skipped = [
        pair for pair in UNSUPPORTED_PAIRS
        if pair[0] in (args.agents or AGENTS) and pair[1] in (args.routes or ROUTES)
    ]
    for harness, route in sorted(skipped):
        print(f"skipped unsupported pair: {harness} × {route}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
