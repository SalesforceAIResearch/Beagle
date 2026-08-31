"""The canonical run-config contract + its drift detector (pydantic ``extra=forbid``).

These lock the contract shape + the detector against drift."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from beagle.agents.core.spec import AgentSource, ModelSpec
from beagle.benchmarks.base import BenchmarkSpec
from beagle.config import (
    AgentConfig,
    AgentSourceConfig,
    BenchmarkConfig,
    ModelConfig,
    RuntimeConfig,
    RunConfig,
    BeagleConfig,
    _load_yaml,
    load_config,
    load_evolve_config,
)
from beagle.rollout.runtime.config import RuntimeConfig as RuntimeSettings
from beagle.types import AgentRole


def _run(**over: object) -> dict:
    base: dict = {
        "model": {"name": "gpt-5.5", "provider": "p"},
        "agent": {"name": "monet", "source": {"repo": "r", "ref": "abc"},
                  "config": {"monet_args": ["--x"]}},
        "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t1", "t2"]},
        "parallelism": 3,
    }
    base.update(over)
    return base


def test_run_config_parses_canonical_shape() -> None:
    rc = RunConfig.from_dict(_run())
    spec = rc.agent_spec()
    assert spec.model and spec.model.name == "gpt-5.5"  # top-level model → the agent
    assert spec.source and spec.source.repo == "r" and spec.source.ref == "abc"
    assert rc.benchmark_spec().task_ids == ["t1", "t2"]
    assert rc.parallelism == 3


def test_detector_rejects_unknown_field() -> None:
    # the invented `limit` (and any typo / renamed / stray field) hard-errors
    with pytest.raises(ValidationError, match="extra_forbidden"):
        BenchmarkConfig(name="b", limit=3)  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RunConfig.from_dict(_run(surprise="x"))


def test_detector_rejects_missing_required() -> None:
    with pytest.raises(ValidationError):  # no benchmark
        RunConfig.from_dict({"model": {"name": "x"}, "agent": {"name": "monet", "config": {}}})


def test_agent_source_inline_form() -> None:
    # repo/ref may live under agent.config.agent_source — honored when `source`
    # is absent, and lifted out of the free-form dict into the typed spec.
    rc = RunConfig.from_dict(_run(agent={"name": "monet",
        "config": {"agent_source": {"repo_url": "https://h/r", "ref": "v1"}}}))
    spec = rc.agent_spec()
    assert spec.source and spec.source.repo == "https://h/r" and spec.source.ref == "v1"
    assert "agent_source" not in spec.config


def test_evolution_config_stamps_roles_and_derives_run() -> None:
    xc = BeagleConfig.from_dict({
        "evolvee": {"name": "monet", "model": {"name": "gpt-5.5"},
                    "source": {"repo": "r", "ref": "main"}},
        "evolver": {"name": "cursor", "model": {"name": "cursor-fast", "provider": "cursor"}},
        "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t1"]},
        "algorithm": "darwinx",  # bare-name form accepted
    })
    assert xc.evolvee.role is AgentRole.EVOLVEE and xc.evolver.role is AgentRole.EVOLVER
    assert xc.algorithm.name == "darwinx"
    derived = xc.run_config()  # evaluate the evolvee: its model becomes the run's model
    assert derived.model.name == "gpt-5.5" and derived.agent.name == "monet"


def test_evolution_config_rejects_swapped_roles() -> None:
    # An explicit role that contradicts the slot is a silent footgun — reject it.
    base = {"evolvee": {"name": "monet"}, "evolver": {"name": "cursor"},
            "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t1"]}}
    with pytest.raises(ValidationError, match="evolvee.role must be 'evolvee'"):
        BeagleConfig.from_dict({**base, "evolvee": {"name": "monet", "role": "evolver"}})
    with pytest.raises(ValidationError, match="evolver.role must be 'evolver'"):
        BeagleConfig.from_dict({**base, "evolver": {"name": "cursor", "role": "evolvee"}})
    # the correct explicit role is accepted (idempotent with the default)
    ok = BeagleConfig.from_dict({**base, "evolvee": {"name": "monet", "role": "evolvee"}})
    assert ok.evolvee.role is AgentRole.EVOLVEE


# ---------------------------------------------------------------------------
# Field-parity guard: if a dataclass field is added without bridging to_spec(),
# the test catches it. Uses structural comparison, not implementation inspection.
# ---------------------------------------------------------------------------


def test_benchmark_config_to_spec_field_parity() -> None:
    """Every BenchmarkConfig field must round-trip through to_spec()."""
    bc = BenchmarkConfig(
        name="terminal_bench_2_1",
        dataset="/data/tb21",
        split="test",
        task_ids=["t1", "t2"],
        exclude_task_ids=["t3"],
        num_samples=2,
        namespace="ns",
        tag="v2",
        registry="reg:5000",
        image="reg/img:tag",
        options={"k": "v"},
    )
    spec = bc.to_spec()
    bc_fields = set(BenchmarkConfig.model_fields.keys())
    bs_fields = {f.name for f in dataclasses.fields(BenchmarkSpec)}
    assert bc_fields == bs_fields, f"BenchmarkConfig/BenchmarkSpec diverged: {bc_fields ^ bs_fields}"
    for field_name in bc_fields:
        assert getattr(spec, field_name) == getattr(bc, field_name), \
            f"Field {field_name!r} not bridged by BenchmarkConfig.to_spec()"


def test_model_config_to_spec_field_parity() -> None:
    """Every ModelConfig field must round-trip through to_spec()."""
    mc = ModelConfig(name="m", provider="p", api_base=None, params={"t": 0.7})
    spec = mc.to_spec()
    mc_fields = set(ModelConfig.model_fields.keys())
    ms_fields = {f.name for f in dataclasses.fields(ModelSpec)}
    assert mc_fields == ms_fields, f"ModelConfig/ModelSpec diverged: {mc_fields ^ ms_fields}"
    for field_name in mc_fields:
        assert getattr(spec, field_name) == getattr(mc, field_name), \
            f"Field {field_name!r} not bridged by ModelConfig.to_spec()"


def test_agent_source_config_to_spec_field_parity() -> None:
    """AgentSourceConfig.to_spec() bridges all declared fields; 'root' is runtime-only."""
    asc = AgentSourceConfig(repo="r", ref="v1", entrypoint="bin/m.js", metadata={"k": "v"})
    spec = asc.to_spec()
    asc_fields = set(AgentSourceConfig.model_fields.keys())
    as_fields = {f.name for f in dataclasses.fields(AgentSource)} - {"root"}  # runtime-only
    assert asc_fields == as_fields, f"AgentSourceConfig/AgentSource diverged: {asc_fields ^ as_fields}"
    for field_name in asc_fields:
        assert getattr(spec, field_name) == getattr(asc, field_name), \
            f"Field {field_name!r} not bridged by AgentSourceConfig.to_spec()"
    assert spec.root is None  # not bridged from config (filled at runtime)


# ---------------------------------------------------------------------------
# RuntimeConfig.to_settings() bridge
# ---------------------------------------------------------------------------


def test_runtime_config_to_settings_passes_through_all_fields() -> None:
    rc = RuntimeConfig(
        kind="xrlenv-cluster",
        grpc_host="h",
        grpc_port=50051,
        token="tok",
        run_id="rid",
        artifact_root="/tmp/art",
        options={"extra": 1},
    )
    s = rc.to_settings()
    assert isinstance(s, RuntimeSettings)
    assert s.kind == "xrlenv-cluster"
    assert s.grpc_host == "h" and s.grpc_port == 50051 and s.token == "tok"
    assert s.run_id == "rid"
    assert s.artifact_root == Path("/tmp/art")  # str → Path conversion
    assert s.options == {"extra": 1}


def test_runtime_config_to_settings_artifact_root_none() -> None:
    """artifact_root=None must NOT be converted to Path(None)."""
    s = RuntimeConfig(kind="local").to_settings()
    assert s.artifact_root is None


# ---------------------------------------------------------------------------
# AgentConfig._resolved_source edge cases
# ---------------------------------------------------------------------------


def test_agent_resolved_source_explicit_wins_over_inline() -> None:
    """When both typed source and inline agent_source are present, typed source wins."""
    ac = AgentConfig(
        name="monet",
        source=AgentSourceConfig(repo="typed-repo", ref="typed-ref"),
        config={"agent_source": {"repo": "inline-repo", "ref": "inline-ref"}},
    )
    resolved = ac._resolved_source()
    assert resolved is not None and resolved.repo == "typed-repo" and resolved.ref == "typed-ref"


def test_agent_resolved_source_inline_repo_key() -> None:
    """Inline form with 'repo' key (not 'repo_url') is resolved."""
    ac = AgentConfig(name="monet", config={"agent_source": {"repo": "https://h/r", "ref": "v1"}})
    resolved = ac._resolved_source()
    assert resolved is not None and resolved.repo == "https://h/r" and resolved.ref == "v1"


def test_agent_resolved_source_inline_repo_url_key() -> None:
    """Inline form with the 'repo_url' key is resolved."""
    ac = AgentConfig(name="monet", config={"agent_source": {"repo_url": "https://h/r2", "ref": "v2"}})
    resolved = ac._resolved_source()
    assert resolved is not None and resolved.repo == "https://h/r2" and resolved.ref == "v2"


def test_agent_resolved_source_non_dict_inline_returns_none() -> None:
    """A non-dict inline agent_source (e.g. a stray string) is silently ignored."""
    ac = AgentConfig(name="monet", config={"agent_source": "not-a-dict"})
    assert ac._resolved_source() is None


def test_agent_resolved_source_none_when_absent() -> None:
    """No source and no inline form → None."""
    assert AgentConfig(name="monet")._resolved_source() is None


# ---------------------------------------------------------------------------
# BeagleConfig.run_config() error paths
# ---------------------------------------------------------------------------


def test_run_config_requires_benchmark() -> None:
    """BeagleConfig.run_config() raises if benchmark is not set."""
    xc = BeagleConfig.from_dict({
        "evolvee": {"name": "monet", "model": {"name": "m"}},
        "evolver": {"name": "cursor", "model": {"name": "c"}},
    })
    with pytest.raises(ValueError, match="benchmark"):
        xc.run_config()


def test_run_config_requires_candidate_model() -> None:
    """BeagleConfig.run_config() raises when the candidate agent has no model."""
    xc = BeagleConfig.from_dict({
        "evolvee": {"name": "monet"},  # no model
        "evolver": {"name": "cursor", "model": {"name": "c"}},
        "benchmark": {"name": "terminal_bench_2_1"},
    })
    with pytest.raises(ValueError, match="no model"):
        xc.run_config()


def test_run_config_evolver_as_candidate() -> None:
    """run_config(agent=evolver) derives a RunConfig for the evolver, not the evolvee."""
    xc = BeagleConfig.from_dict({
        "evolvee": {"name": "monet", "model": {"name": "evolvee-model"}},
        "evolver": {"name": "cursor", "model": {"name": "evolver-model", "provider": "cursor"}},
        "benchmark": {"name": "terminal_bench_2_1"},
    })
    rc = xc.run_config(agent=xc.evolver)
    assert rc.model.name == "evolver-model" and rc.agent.name == "cursor"


# ---------------------------------------------------------------------------
# Detector on nested models: model.params, agent.config pass-through
# ---------------------------------------------------------------------------


def test_detector_nested_model_params_are_freeform() -> None:
    """model.params accepts any dict (freeform escape hatch, not drift-detected)."""
    rc = RunConfig.from_dict(_run(model={"name": "m", "params": {"temperature": 0.7, "max_tokens": 100}}))
    assert rc.model.params == {"temperature": 0.7, "max_tokens": 100}


def test_detector_nested_agent_config_is_freeform() -> None:
    """agent.config accepts any dict (agent-specific knobs, not drift-detected at schema level)."""
    rc = RunConfig.from_dict(_run(agent={"name": "monet",
        "config": {"monet_args": ["--x"], "install_cmd": "npm ci", "unknown_key": True}}))
    assert rc.agent.config["unknown_key"] is True


def test_detector_nested_benchmark_options_are_freeform() -> None:
    """benchmark.options accepts any dict."""
    rc = RunConfig.from_dict(_run(benchmark={"name": "terminal_bench_2_1",
                                             "options": {"parallelism": 4, "timeout": 300}}))
    assert rc.benchmark.options == {"parallelism": 4, "timeout": 300}


# ---------------------------------------------------------------------------
# _load_yaml edge cases
# ---------------------------------------------------------------------------


def test_load_yaml_rejects_list(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="YAML mapping"):
        _load_yaml(f)


def test_load_yaml_empty_file_gives_empty_dict(tmp_path: Path) -> None:
    f = tmp_path / "empty.yaml"
    f.write_text("")
    assert _load_yaml(f) == {}


def test_load_config_from_yaml_file(tmp_path: Path) -> None:
    """load_config() round-trip through a real YAML file."""
    cfg = {
        "model": {"name": "gpt-5.5", "provider": "p"},
        "agent": {"name": "monet", "config": {}},
        "benchmark": {"name": "terminal_bench_2_1"},
    }
    f = tmp_path / "run.yaml"
    f.write_text(yaml.dump(cfg))
    rc = load_config(f)
    assert rc.model.name == "gpt-5.5" and rc.benchmark.name == "terminal_bench_2_1"


def test_load_evolve_config_from_yaml_file(tmp_path: Path) -> None:
    """load_evolve_config() round-trip through a real YAML file."""
    cfg = {
        "evolvee": {"name": "monet", "model": {"name": "m"}},
        "evolver": {"name": "cursor", "model": {"name": "c", "provider": "cursor"}},
        "benchmark": {"name": "terminal_bench_2_1"},
    }
    f = tmp_path / "evolve.yaml"
    f.write_text(yaml.dump(cfg))
    xc = load_evolve_config(f)
    assert xc.evolvee.role is AgentRole.EVOLVEE


# ---------------------------------------------------------------------------
# BenchmarkConfig.num_samples / parallelism validation constraints
# ---------------------------------------------------------------------------


def test_benchmark_num_samples_ge1() -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig(name="b", num_samples=0)


def test_run_config_parallelism_ge1() -> None:
    with pytest.raises(ValidationError):
        RunConfig.from_dict(_run(parallelism=0))


# ---------------------------------------------------------------------------
# Empty task_ids is not explicitly blocked (intentional)
# ---------------------------------------------------------------------------


def test_benchmark_empty_task_ids_accepted() -> None:
    """task_ids=[] is accepted at schema level; it produces zero tasks at runtime.

    This is intentional — no artificial empty-list guard.
    The smoke test's _run_baseline() detects the empty result and raises RuntimeError.
    """
    bc = BenchmarkConfig(name="terminal_bench_2_1", task_ids=[])
    assert bc.task_ids == []


def test_agent_source_token_env_lifts_to_config() -> None:
    # The DarwinX driver / coding-bench nest the clone credential under agent_source; the agent
    # adapter reads TOP-LEVEL config.token_env. to_spec must lift it, else the in-container clone
    # runs unauthenticated ("could not read Username") and the agent never runs. (regression)
    from beagle.config import AgentConfig

    spec = AgentConfig(name="monet", config={
        "agent_source": {"repo": "https://x/r", "ref": "abc", "token_env": "GH_TOKEN"}}).to_spec()
    assert spec.config.get("token_env") == "GH_TOKEN"
    # container_path lifts too — else the clone lands at the adapter default, not where
    # install_cmd `cd`s (a "No such file or directory" install failure).
    spec_cp = AgentConfig(name="monet", config={
        "agent_source": {"repo": "x", "container_path": "/opt/agent"}}).to_spec()
    assert spec_cp.config.get("container_path") == "/opt/agent"
    assert spec.source is not None and spec.source.repo == "https://x/r"
    # an explicit top-level token_env is NOT overwritten
    spec2 = AgentConfig(name="monet", config={
        "token_env": "OTHER", "agent_source": {"repo": "x", "token_env": "GH_TOKEN"}}).to_spec()
    assert spec2.config.get("token_env") == "OTHER"
