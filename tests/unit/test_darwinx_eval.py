"""The DarwinX eval adapter — config translate + the algorithm-shaped run.json serializer.

Hermetic: no cluster, no harbor. The eval is faked; we assert the emitted run.json is
byte-shaped exactly as the vendored algorithm's parser (`codingbench_eval.parse_run_json`)
reads it, and that a faithful re-implementation of that parser recovers the right score."""

from __future__ import annotations

import json
from types import SimpleNamespace

from beagle.algorithms.darwinx import eval as ev
from beagle.types import TaskResult


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
