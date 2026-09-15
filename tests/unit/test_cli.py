"""The `beagle` CLI — arg parsing + dispatch (hermetic; no real run)."""

from __future__ import annotations

from pathlib import Path

import pytest

from beagle import cli
from beagle.config import load_config
from beagle.types import Task, TaskContext

#: A resolved run dir the dry-run pre-flight is handed (from the config's run: block or --run-dir).
_RD = Path("results/runs/RID")


def _items(*task_ids: str, benchmark: str = "terminal_bench_2_1"):
    """Materialized (Task, TaskContext) items, as the Runner/dry-run see them."""
    return [(Task(task_id=i, benchmark=benchmark), TaskContext(image=None)) for i in task_ids]


def test_dry_run_prints_plan_and_rolls_out_nothing(tmp_path, monkeypatch, capsys) -> None:
    """`--dry-run` resolves the plan (run id, tasks, source, gateway pre-flight) and
    returns 0 without touching the Runner — the pre-spend guard for a gate run."""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "model: {name: gpt-5.5}\n"
        "agent:\n  name: monet\n  config:\n"
        "    provider: {type: internal, name: llm-gateway-express-local-proxy}\n"
        "    forward_env: [[LLM_GATEWAY_EXPRESS_API_KEY, HOST_KEY]]\n"
        "benchmark: {name: terminal_bench_2_1, task_ids: [t1, t2]}\n"
        "parallelism: 2\nruntime: {kind: xrlenv-cluster}\n")
    cfg = load_config(str(cfg_path))
    monkeypatch.setenv("HOST_KEY", "present")  # forward_env source resolves → forwarded

    rc = cli._dry_run(cfg, cfg.agent_spec(), _items("t1", "t2"), run_dir=_RD)
    out = capsys.readouterr().out

    assert rc == 0
    assert "DRY RUN" in out and "rolls out NOTHING" in out
    assert "t1" in out and "t2" in out                      # task selection shown
    assert "provider internal (llm-gateway-express-local-proxy)" in out
    assert "1/1 host vars set" in out and "HOST_KEY" in out  # gateway pre-flight
    assert "✓ resolves via benchmarks.get" in out           # Runner-lookup pre-flight
    assert "terminal_bench_2_1" in out and "xrlenv-cluster" in out


@pytest.mark.parametrize("agent_name", ["mini-swe", "opencode"])
def test_dry_run_resolves_explicit_gateway_provider(
    agent_name, tmp_path, monkeypatch, capsys
) -> None:
    cfg_path = tmp_path / "gateway.yaml"
    cfg_path.write_text(
        "model: {name: gpt-5.5}\n"
        f"agent:\n  name: {agent_name}\n  config:\n"
        "    provider:\n"
        "      type: gateway\n"
        "      name: org\n"
        "      extra_args:\n"
        "        api_base: https://gw.example/openai/v1\n"
        "        api_key_env: ORG_API_KEY\n"
        "        auth_header: X-Api-Key\n"
        "benchmark: {name: terminal_bench_2_1, task_ids: [t1]}\n"
    )
    monkeypatch.setenv("ORG_API_KEY", "present")
    cfg = load_config(str(cfg_path))

    rc = cli._dry_run(cfg, cfg.agent_spec(), _items("t1"), run_dir=_RD)
    out = capsys.readouterr().out

    assert rc == 0
    assert "provider    : gateway https://gw.example/openai/v1" in out
    assert "key from $ORG_API_KEY ✓, X-Api-Key header" in out


def test_dry_run_resume_plan_shows_retry_vs_keep(tmp_path, capsys) -> None:
    """`--dry-run --resume [--retry-errors]` prints the per-task plan the Runner would execute:
    each re-run task with its signal, and a kept-by-category summary — glimpsed before any spend."""
    import json

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "model: {name: gpt-5.5}\nagent: {name: monet, config: {}}\n"
        "benchmark: {name: terminal_bench_2_1, task_ids: [t_pass, t_noattempt, t_timeout, t_genuine, t_gone]}\n"
        "runtime: {kind: xrlenv-cluster}\n")
    cfg = load_config(str(cfg_path))
    run_dir = tmp_path / "RID"
    bench = run_dir / "terminal_bench_2_1"

    def _write(tid, reward, *, tokens=(100, 20), exc=None):
        d = bench / tid
        d.mkdir(parents=True)
        rj = {"verifier_result": {"rewards": {"reward": reward}},
              "agent_result": {"n_input_tokens": tokens[0], "n_output_tokens": tokens[1]}}
        if exc:
            rj["exception_info"] = {"exception_type": exc, "exception_message": "x"}
        (d / "result.json").write_text(json.dumps(rj))

    _write("t_pass", 1.0)                          # resolved → keep
    _write("t_noattempt", 0.0, tokens=(0, 0))      # 0 tokens → NoAttempt (error E) → re-run
    _write("t_timeout", 0.0, exc="AgentTimeoutError")   # error E (user's discretion) → re-run
    _write("t_genuine", 0.0)                        # ran, unresolved, no error → genuine-fail F → keep
    # t_gone: no result.json → missing → re-run

    items = _items("t_pass", "t_noattempt", "t_timeout", "t_genuine", "t_gone")
    rc = cli._dry_run(cfg, cfg.agent_spec(), items, run_dir=run_dir, resume=True, retry_errors=True)
    out = capsys.readouterr().out

    assert rc == 0 and "resume plan" in out
    # re-run set: missing + the whole error class E (NoAttempt + timeout); NOT genuine-fail / pass
    assert "↻ RETRY (3)" in out
    assert "t_gone" in out and "t_noattempt" in out and "NoAttempt" in out and "t_timeout" in out
    assert "→ 3 re-run, 2 kept" in out
    # kept, summarized by category — only the resolved pass and the genuine capability failure
    assert "resolved 1" in out and "genuine-fail 1" in out
    # cost estimate reflects the RE-RUN count (3), not the total (5)
    assert "× 3 re-run tasks (of 5)" in out and "3 trial containers" in out


def test_dry_run_flags_unresolvable_benchmark(tmp_path, capsys) -> None:
    """The guard that would have caught the cache-name-in-identity bug: if a task's
    benchmark doesn't resolve the way the Runner looks it up, the pre-flight warns
    instead of the live run dying with KeyError mid-spend."""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("model: {name: gpt-5.5}\nagent: {name: monet, config: {}}\n"
                        "benchmark: {name: terminal_bench_2_1, task_ids: [t1]}\n")
    cfg = load_config(str(cfg_path))
    # Simulate the bug: Task.benchmark carries the hyphenated cache name, not the registry name.
    cli._dry_run(cfg, cfg.agent_spec(), _items("t1", benchmark="terminal-bench-2-1"), run_dir=_RD)
    out = capsys.readouterr().out
    assert "⚠" in out and "'terminal-bench-2-1' does NOT resolve" in out


def test_dry_run_partial_forward_env_is_not_flagged(tmp_path, monkeypatch, capsys) -> None:
    """One-of-two alternative creds set (e.g. API_KEY_LIST present, API_KEY absent) is
    fine — the gateway needs ONE, not both. The unset one is reported, not warned."""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "model: {name: gpt-5.5}\n"
        "agent: {name: monet, config: {forward_env: "
        "[[K_LIST, HOST_LIST], [K, HOST_KEY], [URL, HOST_URL]]}}\n"
        "benchmark: {name: terminal_bench_2_1, task_ids: [t1]}\n")
    cfg = load_config(str(cfg_path))
    monkeypatch.setenv("HOST_LIST", "x"); monkeypatch.setenv("HOST_URL", "u")
    monkeypatch.delenv("HOST_KEY", raising=False)  # the alternative — legitimately unset

    cli._dry_run(cfg, cfg.agent_spec(), _items("t1"), run_dir=_RD)
    out = capsys.readouterr().out
    assert "2/3 host vars set" in out                        # partial ≠ problem
    assert "agent gets no creds" not in out                  # NOT the zero-cred warning
    assert "not set : HOST_KEY" in out and "skipped" in out  # reported, not alarmed


def test_dry_run_zero_forwarded_creds_warns(tmp_path, monkeypatch, capsys) -> None:
    """The real footgun: NOTHING forwarded → agent gets no creds (the run-2 failure)."""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "model: {name: gpt-5.5}\n"
        "agent: {name: monet, config: {forward_env: [[K, HOST_KEY], [URL, HOST_URL]]}}\n"
        "benchmark: {name: terminal_bench_2_1, task_ids: [t1]}\n")
    cfg = load_config(str(cfg_path))
    monkeypatch.delenv("HOST_KEY", raising=False); monkeypatch.delenv("HOST_URL", raising=False)

    cli._dry_run(cfg, cfg.agent_spec(), _items("t1"), run_dir=_RD)
    out = capsys.readouterr().out
    assert "⚠" in out and "0/2 host vars set" in out and "agent gets no creds" in out


def test_dry_run_prints_the_resolved_run_dir(tmp_path, capsys) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("model: {name: gpt-5.5}\nagent: {name: monet, config: {}}\n"
                        "benchmark: {name: terminal_bench_2_1, task_ids: [t1]}\n")
    cfg = load_config(str(cfg_path))
    cli._dry_run(cfg, cfg.agent_spec(), _items("t1"), run_dir=Path("/tmp/my-gate-out"))
    out = capsys.readouterr().out
    assert "/tmp/my-gate-out/" in out         # the run dir handed in is what's shown


def test_evolve_requires_data_to_score_on(tmp_path, monkeypatch) -> None:
    # `evolve` takes a canonical --config; a config with a valid evolvee/evolver but no `data`
    # (the benchmark to score candidates on) fails loud before any spend.
    monkeypatch.setattr("beagle.dotenv.load_project_dotenv", lambda *a, **k: None)
    cfg = tmp_path / "e.yaml"
    cfg.write_text(
        "run: {name: e}\n"
        "evolvee: {harness: {name: monet, source: {repo: x, ref: y}}, model: {name: m}}\n"
        "evolver: {harness: {name: cursor}, model: {name: m}}\n"
        "algorithm: {name: darwinx}\n")   # no `data`
    with pytest.raises(ValueError, match="no `data`"):
        cli.main(["evolve", "--config", str(cfg)])


# --- canonical config.yaml loader + evolve/evaluate dispatch ------------------

_CANONICAL_EVAL = """
run: {dir: ./tmp, name: e1, runtime: local, parallelism: 2, timestamp: false}
agent:
  harness:
    name: monet
    version: v1
    source: {repo: https://x/r, ref: abc, token_env: GH_TOKEN, container_path: /opt/agent}
  model: {name: gpt-5.5}
  provider: {type: internal, name: llm-gateway-express-local-proxy}
  effort: high
  max_turns: 150
  forward_env: [LLM_GATEWAY_EXPRESS_API_KEY]
  timeout: 1800
  extra_args:
    monet_args: [--no-monet-md, --output-format, stream-json]
data:
  - {benchmark: terminal_bench_2_1, tasks: [t1, t2]}
"""


def test_canonical_evaluation_preserves_runtime_options(tmp_path):
    from beagle.cli._canonical import build_evaluation, load

    p = tmp_path / "eval.yaml"
    p.write_text(_CANONICAL_EVAL.replace(
        "runtime: local",
        "runtime: {kind: kata, options: {docker_host: 'unix:///run/lab.sock'}}"))
    cfg, _ = build_evaluation(load(p))
    assert cfg.runtime_settings().kind == "kata"
    assert cfg.runtime_settings().options == {"docker_host": "unix:///run/lab.sock"}


def test_evaluate_cli_passes_kata_socket_to_runtime(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import beagle as bgl
    from beagle.rollout.runtime import KataDockerRuntime

    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(_CANONICAL_EVAL.replace(
        "runtime: local",
        "runtime: {kind: kata, options: {docker_host: 'unix:///run/lab.sock'}}"))
    monkeypatch.setattr("beagle.cli._canonical.check_versions", lambda *a, **kw: None)
    monkeypatch.setattr(bgl.TaskDataset, "from_benchmark", lambda *a: [])
    monkeypatch.setattr(bgl.agents, "build", lambda spec: object())
    seen = {}

    def evaluate(config, **kwargs):
        seen["runtime"] = kwargs["runtime"]
        return SimpleNamespace(artifact_dir=tmp_path, score=0.0, results=[])

    monkeypatch.setattr(bgl, "evaluate", evaluate)
    assert cli.main(["evaluate", "--config", str(cfg_path)]) == 0
    assert isinstance(seen["runtime"], KataDockerRuntime)
    assert seen["runtime"].docker_host == "unix:///run/lab.sock"


def test_canonical_build_evaluation(tmp_path) -> None:
    from beagle.cli._canonical import build_evaluation, load
    p = tmp_path / "eval.yaml"; p.write_text(_CANONICAL_EVAL)
    cfg, run_dir = build_evaluation(load(p))
    assert cfg.agent.name == "monet" and cfg.model.name == "gpt-5.5"
    assert cfg.benchmark.name == "terminal_bench_2_1" and cfg.benchmark.task_ids == ["t1", "t2"]
    # first-level vocabulary + the agent's own `extra_args.monet_args` all land flat in agent.config
    ac = cfg.agent.config
    assert ac["provider"] == {
        "type": "internal", "name": "llm-gateway-express-local-proxy"}
    assert ac["effort"] == "high" and ac["max_turns"] == 150
    assert ac["token_env"] == "GH_TOKEN" and ac["timeout"] == 1800
    assert ac["monet_args"] == ["--no-monet-md", "--output-format", "stream-json"]
    assert cfg.runtime.kind == "local" and cfg.parallelism == 2
    assert run_dir.as_posix().endswith("tmp/e1")


def test_run_dir_timestamps_evaluate_but_pins_evolve(monkeypatch) -> None:
    # `evaluate` defaults to a FRESH `<dir>/<name>-<stamp>` per run (so re-running an edited config
    # never hits the "job dir already exists / can't resume with a different config" collision);
    # `run.timestamp: false` pins a stable, resumable dir; `evolve` defaults to a stable campaign dir.
    import beagle.cli._canonical as cn
    monkeypatch.setattr(cn, "_now_stamp", lambda: "20260811-143022")
    d, n = cn._run_dir({"run": {"dir": "./out", "name": "e1"}}, default_timestamp=True)
    assert n == "e1-20260811-143022" and d.as_posix() == "out/e1-20260811-143022"
    d2, n2 = cn._run_dir({"run": {"dir": "./out", "name": "e1", "timestamp": False}},
                         default_timestamp=True)
    assert n2 == "e1" and d2.as_posix() == "out/e1"
    _, n3 = cn._run_dir({"run": {"name": "camp"}}, default_timestamp=False)  # evolve: stable
    assert n3 == "camp"


def test_benchmark_dict_omit_tasks_is_whole_suite() -> None:
    # Omitting `tasks` leaves task_ids unset → BenchmarkConfig default None = the whole suite (what
    # the examples/evaluation configs rely on to benchmark a full benchmark); a list restricts+orders.
    from beagle.cli._canonical import benchmark_dict
    assert "task_ids" not in benchmark_dict({"benchmark": "deep-swe"})
    assert benchmark_dict({"benchmark": "x", "tasks": ["a", "b"]})["task_ids"] == ["a", "b"]


def test_canonical_first_level_vocab_and_per_agent_extra_args() -> None:
    # provider/effort/max_turns are FIRST-LEVEL for every agent; each agent's own args live under
    # `extra_args:` keyed by `<agent>_args` (monet_args / mini_swe_args). Both fold into flat config.
    from beagle.cli._canonical import agent_dict
    monet = agent_dict({
        "harness": {"name": "monet", "source": {"repo": "r", "token_env": "GH_TOKEN"}},
        "model": {"name": "m"}, "forward_env": ["A"], "timeout": 1800,
        "provider": {"type": "internal", "name": "gw"},
        "effort": "high", "max_turns": 150,
        "extra_args": {"monet_args": ["--x"]}})           # monet's own args = a raw CLI list
    c = monet["config"]
    assert c["provider"] == {"type": "internal", "name": "gw"}
    assert c["effort"] == "high" and c["max_turns"] == 150
    assert c["monet_args"] == ["--x"]                                                  # extra_args
    assert c["forward_env"] == ["A"] and c["timeout"] == 1800 and c["token_env"] == "GH_TOKEN"

    # mini-swe: SAME first-level vocab; its own args are named knobs (config_path) under mini_swe_args
    # — written as the list-of-one-map form and flattened into config.
    mini = agent_dict({"harness": {"name": "mini-swe"}, "model": {"name": "m"},
                       "provider": {"type": "internal", "name": "gw"},
                       "effort": "high", "max_turns": 150,
                       "extra_args": {"mini_swe_args": [{"config_path": "mini.yaml"}]}})
    mc = mini["config"]
    assert mc["provider"] == {"type": "internal", "name": "gw"}
    assert mc["effort"] == "high" and mc["max_turns"] == 150
    assert mc["config_path"] == "mini.yaml"

    # a plain map under <agent>_args works too; and flat top-level knobs still fold (backward-compat).
    m2 = agent_dict({"harness": {"name": "mini-swe"}, "model": {"name": "m"},
                     "extra_args": {"mini_swe_args": {"config_path": "b.yaml"}}})
    assert m2["config"]["config_path"] == "b.yaml"
    legacy = agent_dict({"harness": {"name": "monet"}, "model": {"name": "m"},
                         "monet_args": ["--a"],
                         "provider": {"type": "internal", "name": "gw"}})
    assert legacy["config"]["monet_args"] == ["--a"]
    assert legacy["config"]["provider"] == {"type": "internal", "name": "gw"}

    # source.entrypoint (the invoke/config path — e.g. mini's config YAML) is carried through;
    # dropping it made a repo+ref yaml install mini with an empty entrypoint → `-c /agent/` (#12).
    with_src = agent_dict({"harness": {"name": "mini-swe",
                                       "source": {"repo": "r", "ref": "abc",
                                                  "entrypoint": "path/to/x.yaml"}},
                           "model": {"name": "m"}})
    assert with_src["source"] == {"repo": "r", "ref": "abc", "entrypoint": "path/to/x.yaml"}


def test_canonical_folds_prompt_override(tmp_path) -> None:
    # The optional layer-1/2 override escape hatch folds from the role block into agent.config,
    # verbatim, so a config-driven adapter (mini-swe) can apply it.
    from beagle.cli._canonical import build_evaluation, load
    text = _CANONICAL_EVAL.replace(
        "data:\n", "  prompt_override:\n    system: SYS-P\n    instruction: 'do {{task}}'\ndata:\n")
    p = tmp_path / "ov.yaml"; p.write_text(text)
    cfg, _ = build_evaluation(load(p))
    assert cfg.agent.config["prompt_override"] == {"system": "SYS-P", "instruction": "do {{task}}"}


def test_canonical_folds_the_gateway_provider(tmp_path) -> None:
    # An org's OpenAI-compatible proxy (its own api_base + its own key var). It folds into
    # agent.config — NOT the model block — because agent.config is the only block the harbor shim
    # preserves, so the routing survives to the in-trial adapter.
    from beagle.agents.core.litellm_gateway import resolve_gateway
    from beagle.cli._canonical import build_evaluation, load
    text = _CANONICAL_EVAL.replace(
        "data:\n", "  provider:\n    type: gateway\n    name: org\n    extra_args:\n"
                   "      api_base: https://gw.example/openai/v1\n"
                   "      api_key_env: MY_ORG_API_KEY\n      auth_header: x-api-key\ndata:\n")
    p = tmp_path / "gw.yaml"; p.write_text(text)
    cfg, _ = build_evaluation(load(p))
    assert cfg.agent.config["provider"] == {
        "type": "gateway", "name": "org", "extra_args": {
            "api_base": "https://gw.example/openai/v1",
            "api_key_env": "MY_ORG_API_KEY", "auth_header": "x-api-key"}}
    assert resolve_gateway(cfg.agent.config)["api_base"] == "https://gw.example/openai/v1"


def test_canonical_rejects_old_and_malformed_provider_shapes(tmp_path) -> None:
    from beagle.cli._canonical import build_evaluation, load
    text = _CANONICAL_EVAL.replace("data:\n", '  gateway: ""\ndata:\n')
    p = tmp_path / "bad.yaml"; p.write_text(text)
    with pytest.raises(ValueError, match="top-level `gateway:` was replaced"):
        build_evaluation(load(p))

    p.write_text(_CANONICAL_EVAL.replace(
        "provider: {type: internal, name: llm-gateway-express-local-proxy}",
        "provider: old-scalar"))
    with pytest.raises(ValueError, match="scalar `provider:` was replaced"):
        build_evaluation(load(p))


def test_canonical_old_agent_key_gives_migration_error(tmp_path) -> None:
    # The nested block was renamed `agent:` → `harness:`; a config still using the old key gets a
    # clear migration error (not a cryptic KeyError on the missing name).
    from beagle.cli._canonical import build_evaluation, load
    p = tmp_path / "old.yaml"
    p.write_text("run: {name: e}\nagent:\n  agent: {name: monet}\n  model: {name: m}\n"
                 "data:\n  - {benchmark: b, tasks: [t]}\n")
    with pytest.raises(ValueError, match="renamed to `harness"):
        build_evaluation(load(p))


def test_evaluate_and_evolve_dispatch(monkeypatch, tmp_path) -> None:
    from beagle import cli
    seen = {}
    monkeypatch.setattr("beagle.cli.evaluate._cmd_evaluate", lambda a: seen.update(cmd="evaluate", cfg=a.config, dry=a.dry_run) or 0)
    monkeypatch.setattr("beagle.cli.evolve._cmd_evolve", lambda a: seen.update(cmd="evolve", cfg=a.config, dry=a.dry_run) or 0)
    monkeypatch.setattr("beagle.dotenv.load_project_dotenv", lambda *a, **k: None)
    assert cli.main(["evaluate", "--config", "c.yaml", "--dry-run"]) == 0
    assert seen == {"cmd": "evaluate", "cfg": "c.yaml", "dry": True}
    assert cli.main(["evolve", "--config", "c.yaml"]) == 0            # default = run (no --dry-run)
    assert seen == {"cmd": "evolve", "cfg": "c.yaml", "dry": False}


def test_evaluate_parses_the_run_ops_flags(monkeypatch) -> None:
    # `evaluate` carries the run ops flags folded in from the retired `run` command.
    seen = {}
    monkeypatch.setattr("beagle.cli.evaluate._cmd_evaluate", lambda a: seen.update(
        resume=a.resume, retry=a.retry_errors, force=a.force_resume,
        campaign=a.campaign_id, run_id=a.run_id, run_dir=a.run_dir) or 0)
    monkeypatch.setattr("beagle.dotenv.load_project_dotenv", lambda *a, **k: None)
    rc = cli.main(["evaluate", "--config", "c.yaml", "--resume", "--retry-errors", "--force-resume",
                   "--campaign-id", "camp", "--run-id", "RID", "--run-dir", "/tmp/d",
                   "--task-ids", "t1,t2"])
    assert rc == 0
    assert seen == {"resume": True, "retry": True, "force": True,
                    "campaign": "camp", "run_id": "RID", "run_dir": "/tmp/d"}


def test_retry_flags_are_independent_and_task_ids_restricts_not_filters(monkeypatch, tmp_path) -> None:
    """The three resume/retry flags are independent (no implication), and --task-ids is a re-run
    RESTRICTION, not a dataset filter: the FULL dataset still reaches the runner (so run.json covers
    the whole benchmark) while ``only_task_ids`` narrows what actually re-runs."""
    from beagle import cli
    from beagle.data.dataset import TaskDataset

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_CANONICAL_EVAL)
    ds = TaskDataset(_items("t1", "t2", "t3"), name="terminal_bench_2_1")
    monkeypatch.setattr("beagle.TaskDataset.from_benchmark", lambda spec: ds)
    monkeypatch.setattr("beagle.dotenv.load_project_dotenv", lambda *a, **k: None)
    cap: dict = {}
    monkeypatch.setattr("beagle.cli.evaluate._dry_run",
                        lambda cfg, spec, items, **kw: cap.update(
                            ids=[t.task_id for t, _ in items], resume=kw["resume"],
                            retry_errors=kw["retry_errors"], only=kw.get("only_task_ids")) or 0)

    # --retry-errors WITHOUT --resume, scoped to t1,t3
    rc = cli.main(["evaluate", "--config", str(cfg_path), "--dry-run", "--retry-errors",
                   "--task-ids", "t1,t3"])
    assert rc == 0
    assert cap["resume"] is False             # NOT implied — independent flags
    assert cap["retry_errors"] is True
    assert cap["ids"] == ["t1", "t2", "t3"]   # FULL dataset reaches the runner (run.json stays whole)
    assert cap["only"] == {"t1", "t3"}        # --task-ids only RESTRICTS the re-run set


def test_task_ids_unknown_fails_loud(monkeypatch, tmp_path) -> None:
    from beagle import cli
    from beagle.data.dataset import TaskDataset

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_CANONICAL_EVAL)
    monkeypatch.setattr("beagle.TaskDataset.from_benchmark",
                        lambda spec: TaskDataset(_items("t1"), name="terminal_bench_2_1"))
    monkeypatch.setattr("beagle.dotenv.load_project_dotenv", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="unknown task id"):
        cli.main(["evaluate", "--config", str(cfg_path), "--dry-run", "--task-ids", "nope"])


def test_main_keyboard_interrupt_exits_cleanly(monkeypatch, capsys, tmp_path) -> None:
    # Ctrl-C during a run already tears the cluster containers down (stop_run_on_sigint); main() must
    # exit with the conventional 130 + a one-liner, not dump a KeyboardInterrupt traceback at the user.
    import beagle.cli.evaluate as ev
    from beagle import cli

    def _raise(_args):
        raise KeyboardInterrupt

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_CANONICAL_EVAL)
    monkeypatch.setattr("beagle.dotenv.load_project_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(ev, "_cmd_evaluate", _raise)

    rc = cli.main(["evaluate", "--config", str(cfg_path)])
    assert rc == 130
    assert "aborted (SIGINT)" in capsys.readouterr().err
