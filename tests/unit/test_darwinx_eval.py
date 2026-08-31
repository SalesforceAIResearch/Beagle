"""The DarwinX eval adapter — config translate + the algorithm-shaped run.json serializer.

Hermetic: no cluster, no harbor. The eval is faked; we assert the emitted run.json is
byte-shaped exactly as the vendored algorithm's parser (`codingbench_eval.parse_run_json`)
reads it, and that a faithful re-implementation of that parser recovers the right score."""

from __future__ import annotations

import json
from types import SimpleNamespace

from beagle.algorithms.darwinx import _launch
from beagle.algorithms.darwinx import eval as ev
from beagle.types import TaskResult


def _cbe():
    """The vendored eval module (needs the vendored import path prepared, like the pipeline)."""
    _launch.prepare_import_path()
    from evolve import codingbench_eval  # noqa: PLC0415 — import after path prep

    return codingbench_eval


def test_translate_config_renames_token_and_drops_grpc_secure() -> None:
    cfg = ev.translate_config({
        "model": {"name": "gpt-5.5"},
        "agent": {"name": "monet", "config": {"agent_source": {"repo": "r", "ref": "b"}}},
        "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t1"]},
        "runtime": {"kind": "xrlenv-cluster", "consumer_token": "TOK", "grpc_secure": True},
        "parallelism": 2,
    })
    assert cfg.runtime.token == "TOK"          # consumer_token → token
    assert cfg.runtime.kind == "xrlenv-cluster"  # grpc_secure dropped, no error
    assert cfg.benchmark.task_ids == ["t1"] and cfg.parallelism == 2


def test_translate_config_drops_coding_bench_dataset_ref() -> None:
    # the algorithm's `benchmark.dataset` is a coding-bench path ref; beagle loads harbor tasks
    # from $XRLENV_BENCHMARK_CACHE, and BenchmarkSpec.dataset is a task-source path — dropping it
    # keeps the eval reading the cache (task_ids/name unchanged).
    cfg = ev.translate_config({
        "model": {"name": "gpt-5.5"}, "agent": {"name": "monet"},
        "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["adaptive-rejection-sampler"],
                      "dataset": "benchmarks/terminal_bench/vendor"},
        "runtime": {"kind": "local"},
    })
    assert cfg.benchmark.dataset is None
    assert cfg.benchmark.name == "terminal_bench_2_1"
    assert cfg.benchmark.task_ids == ["adaptive-rejection-sampler"]


def _results() -> list[TaskResult]:
    return [
        TaskResult(task_id="bn-fit-modify__ABC", resolved=True, reward=1.0,
                   tokens={"prompt": 100, "completion": 20}),
        TaskResult(task_id="adaptive-rejection-sampler__XY", resolved=False, reward=0.0,
                   tokens={"prompt": 50, "completion": 5}),
        TaskResult(task_id="broken__ZZ", resolved=False, reward=None,
                   error="AgentTimeoutError: deadline exceeded"),
    ]


def test_to_darwinx_run_json_matches_contract() -> None:
    d = ev.to_darwinx_run_json(_results())

    # Rows keyed by BASE task id (harbor __<hash> suffix stripped), with the exact keys the
    # algorithm reads.
    rows = d["per_task_results"]
    assert [r["task_id"] for r in rows] == ["bn-fit-modify", "adaptive-rejection-sampler", "broken"]
    assert rows[0] == {"task_id": "bn-fit-modify", "resolved": True, "reward": 1.0,
                       "error": None, "tokens": {"prompt": 100, "completion": 20}}
    assert rows[2]["reward"] is None and rows[2]["error"].startswith("AgentTimeoutError")

    # errors[] carries the full, substring-matchable message (contract #2).
    assert d["errors"] == [{"task_id": "broken", "kind": "error",
                            "message": "AgentTimeoutError: deadline exceeded", "traceback": ""}]

    # Totals: reward falls back to 1.0-if-resolved; errored counted.
    assert d["totals"] == {"num_tasks": 3, "num_tasks_resolved": 1, "num_tasks_errored": 1}


def _parser_score(run_json: dict) -> tuple[float, int]:
    """A faithful re-impl of the algorithm's scoring core (mean effective reward + resolved
    count) — proves our run.json yields the numbers its real parser would compute."""
    rows = run_json.get("per_task_results") or []
    eff = [r["reward"] if r.get("reward") is not None else (1.0 if r.get("resolved") else 0.0)
           for r in rows]
    score = sum(eff) / len(eff) if eff else 0.0
    return score, sum(1 for e in eff if e >= 1.0)


def test_round_trip_parser_recovers_score() -> None:
    score, resolved = _parser_score(ev.to_darwinx_run_json(_results()))
    assert score == 1 / 3 and resolved == 1


def test_write_preserves_clean_summary_beside_compat(tmp_path) -> None:
    (tmp_path / "run.json").write_text('{"benchmarks": {}, "totals": {}}')  # beagle's clean shape
    ev.write_darwinx_run_json(tmp_path, _results())

    compat = json.loads((tmp_path / "run.json").read_text())
    assert "per_task_results" in compat                     # run.json is now the algorithm shape
    clean = json.loads((tmp_path / "run.beagle.json").read_text())
    assert "benchmarks" in clean                            # clean summary preserved beside it


def test_run_eval_writes_compat_run_json_at_discovered_path(tmp_path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "model: {name: gpt-5.5}\n"
        "agent: {name: monet, config: {agent_source: {repo: r, ref: b}}}\n"
        "benchmark: {name: terminal_bench_2_1, task_ids: [old]}\n"
        "runtime: {kind: xrlenv-cluster, consumer_token: TOK}\nparallelism: 2\n")

    seen: dict = {}

    def _fake_evaluate(config, *, run_id, run_dir, campaign_id):  # noqa: ANN001 — the eval seam
        seen["task_ids"] = list(config.benchmark.task_ids)
        seen["campaign_id"] = campaign_id
        seen["parallelism"] = config.parallelism
        from pathlib import Path
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "run.json").write_text('{"benchmarks": {}}')  # the clean run.json
        return SimpleNamespace(results=_results())

    run_dir = ev.run_eval(
        cfg_path, results_root=tmp_path / "out", run_id="RID",
        include_task_name=["bn-fit-modify"], campaign_id="camp", _evaluate=_fake_evaluate,
    )

    # Discovered at <results-root>/runs/<run_id>/ (the layout the algorithm snapshots).
    assert run_dir == tmp_path / "out" / "runs" / "RID"
    d = json.loads((run_dir / "run.json").read_text())
    assert _parser_score(d) == (1 / 3, 1)                    # the algorithm gets the right score
    assert (run_dir / "run.beagle.json").exists()           # clean summary preserved
    # --include-task-name overrode task_ids; parallelism + campaign threaded through to eval.
    assert seen == {"parallelism": 2, "task_ids": ["bn-fit-modify"], "campaign_id": "camp"}


# --- evolvee-agnostic agent block (the general bridge) -----------------------

def test_agent_block_monet_default_is_unchanged() -> None:
    # Default evolvee (name "monet") → the fully-wired monet block: install_cmd + monet_args +
    # the monet-shaped agent_source (with container_path). This is the byte-identical legacy path.
    cbe = _cbe()
    block = cbe._agent_block(cbe.CodingBenchEvalConfig(), forward_env=[{"K": "K"}])
    assert block["name"] == "monet"
    cfg = block["config"]
    assert "install_cmd" in cfg and cfg["monet_args"][0] == "--provider"
    assert cfg["agent_source"]["container_path"] == "/opt/agent"
    assert cfg["forward_env"] == [{"K": "K"}]


def test_agent_block_non_monet_emits_general_bridge() -> None:
    # A mini-swe evolvee → the lean {name, config} bridge: NO monet install_cmd / monet_args, the
    # evolvee's own knobs threaded verbatim, budgets + code-version defaulted in.
    import dataclasses

    cbe = _cbe()
    cb_cfg = dataclasses.replace(
        cbe.CodingBenchEvalConfig(),
        eval_agent_name="mini-swe",
        eval_agent_config={"provider": "sfr-gateway", "effort": "high",
                           "config_path": "src/minisweagent/config/benchmarks/swebench.yaml"},
        monet_ref="cand-branch", max_turns=25, timeout=1200,
    )
    block = cbe._agent_block(cb_cfg, forward_env=[{"K": "K"}])
    assert block["name"] == "mini-swe"
    cfg = block["config"]
    assert "install_cmd" not in cfg and "monet_args" not in cfg          # NOT monet-shaped
    assert cfg["provider"] == "sfr-gateway" and cfg["effort"] == "high"  # evolvee knobs threaded
    assert cfg["config_path"] == "src/minisweagent/config/benchmarks/swebench.yaml"
    assert cfg["max_turns"] == 25 and cfg["timeout"] == 1200            # budgets defaulted in
    assert cfg["forward_env"] == [{"K": "K"}]                            # cred forwarding defaulted
    src = cfg["agent_source"]
    assert src["ref"] == "cand-branch" and src["repo_url"] == cb_cfg.monet_repo_url


def test_agent_block_non_monet_explicit_knobs_win() -> None:
    # Explicit forward_env / budgets in the evolvee config override the bridge defaults.
    import dataclasses

    cbe = _cbe()
    cb_cfg = dataclasses.replace(
        cbe.CodingBenchEvalConfig(), eval_agent_name="mini-swe", max_turns=25,
        eval_agent_config={"forward_env": [{"OWN": "OWN"}], "max_turns": 99, "token_env": "MY_TOK"})
    cfg = cbe._agent_block(cb_cfg, forward_env=[{"K": "K"}])["config"]
    assert cfg["forward_env"] == [{"OWN": "OWN"}] and cfg["max_turns"] == 99
    assert cfg["agent_source"]["token_env"] == "MY_TOK"                  # evolvee token_env wins


def test_build_codingbench_config_flips_agent_name_for_non_monet() -> None:
    cbe = _cbe()
    import dataclasses

    cb_cfg = dataclasses.replace(cbe.CodingBenchEvalConfig(), eval_agent_name="mini-swe",
                                 runtime_kind="local")
    doc = cbe.build_codingbench_config(task_names=["t1"], cb_cfg=cb_cfg)
    assert doc["agent"]["name"] == "mini-swe"
    assert "install_cmd" not in doc["agent"]["config"]


def test_from_self_evolve_config_reads_name_and_config(tmp_path) -> None:
    # The campaign's benchmarked-agent block carries name + config (emitted by _launch) → the eval
    # config picks up the evolvee selector + its knobs; a legacy block (no name) stays monet.
    cbe = _cbe()
    cfg_path = tmp_path / "camp.yaml"
    cfg_path.write_text(
        "monet:\n  name: mini-swe\n  model: gpt-5.5\n  max_turns: 30\n"
        "  config: {provider: sfr-gateway, effort: high}\n"
        "runtime: {kind: local}\n")
    cb_cfg = cbe.CodingBenchEvalConfig.from_self_evolve_config(cfg_path)
    assert cb_cfg.eval_agent_name == "mini-swe"
    assert cb_cfg.eval_agent_config == {"provider": "sfr-gateway", "effort": "high"}
    assert cb_cfg.model_name == "gpt-5.5" and cb_cfg.max_turns == 30

    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("monet: {model: gpt-5.5}\nruntime: {kind: local}\n")
    assert cbe.CodingBenchEvalConfig.from_self_evolve_config(legacy).eval_agent_name == "monet"


def test_end_to_end_mini_swe_evolvee_resolves_to_registry_agent(tmp_path) -> None:
    # The WHOLE bridge, hermetic (no spend): a mini-swe evolvee → emit_campaign_config →
    # from_self_evolve_config → build_codingbench_config → translate_config → RunConfig →
    # beagle.agents.build resolves to the REAL MiniSweAgent, carrying the evolvee's code version
    # + its own adapter knobs. This is the seam that pins the evolvee to a registry NAME, not monet.
    import beagle.agents as bagents
    from beagle.agents.core.spec import AgentSource, AgentSpec, ModelSpec
    from beagle.agents.mini_swe import MiniSweAgent
    from beagle.algorithms.darwinx import _launch
    from beagle.config import RunConfig

    cbe = _cbe()
    evolvee = bagents.build(AgentSpec(
        name="mini-swe", model=ModelSpec(name="gpt-5.5"),
        source=AgentSource(repo="https://example.test/mini-copy", ref="cand-branch"),
        config={"provider": "sfr-gateway", "effort": "high", "max_turns": 30, "timeout": 1800,
                "config_path": "src/minisweagent/config/benchmarks/swebench.yaml",
                "token_env": "GH_TOKEN"}))
    evolver = SimpleNamespace(spec=AgentSpec(name="cursor", model=ModelSpec(name="auto")))
    run_cfg = RunConfig.from_dict({
        "model": {"name": "gpt-5.5"}, "agent": {"name": "mini-swe"},
        "benchmark": {"name": "terminal_bench_2_1"}, "runtime": {"kind": "local"}})

    camp = _launch.emit_campaign_config(evolvee=evolvee, evolver=evolver,
                                        run_config=run_cfg, dest=tmp_path / "camp.yaml")
    cb_cfg = cbe.CodingBenchEvalConfig.from_self_evolve_config(camp)
    assert cb_cfg.eval_agent_name == "mini-swe"

    doc = cbe.build_codingbench_config(task_names=["t1"], cb_cfg=cb_cfg)
    assert doc["agent"]["name"] == "mini-swe" and "install_cmd" not in doc["agent"]["config"]

    spec = ev.translate_config(doc).agent_spec()
    built = bagents.build(spec)
    assert isinstance(built, MiniSweAgent)                                # resolved via the registry
    assert built.source().repo == "https://example.test/mini-copy"       # evolvee θ (code version)
    assert built.source().ref == "cand-branch"
    assert built.config.get("provider") == "sfr-gateway"                 # adapter knobs survived
    assert built.config.get("config_path") == "src/minisweagent/config/benchmarks/swebench.yaml"
    assert built.config.get("token_env") == "GH_TOKEN"
    assert built.config.get("max_turns") == 30                           # budget threaded through


def test_run_eval_tags_acquires_with_group_id(tmp_path) -> None:
    # The group scope must be ACTIVE while do_eval runs, so every container the candidate's eval
    # acquires (harbor/swebench go through the drop-in, inheriting the contextvar) carries
    # xrlenv.group_id=run_id — the tag a Ctrl-C teardown targets. Restored after run_eval returns.
    from xrlenv.compat.metadata import current_rollout_metadata

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "model: {name: gpt-5.5}\n"
        "agent: {name: monet, config: {agent_source: {repo: r, ref: b}}}\n"
        "benchmark: {name: terminal_bench_2_1, task_ids: [t]}\n"
        "runtime: {kind: xrlenv-cluster, consumer_token: TOK}\n")

    seen: dict = {}

    def _fake_evaluate(config, *, run_id, run_dir, campaign_id):  # noqa: ANN001
        seen["group_id"] = current_rollout_metadata().group_id  # active inside do_eval
        from pathlib import Path
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(results=_results())

    ev.run_eval(cfg_path, results_root=tmp_path / "out", run_id="RID",
                campaign_id="camp", _evaluate=_fake_evaluate)
    assert seen["group_id"] == "RID"                          # tagged with the run's group
    assert current_rollout_metadata().group_id is None        # scope restored on exit
