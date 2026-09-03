"""The unified Runner — group-by-benchmark batch rollout → grade → run.json + resume.

Hermetic: a fake benchmark (canned harness + reducing grader) is injected via
``benchmarks.get``; no cluster, no harbor, no Docker."""

from __future__ import annotations

import json

import pytest

import beagle.benchmarks as benchmarks
from beagle.benchmarks.base import GradeReport
from beagle.config import RunConfig
from beagle.rollout.runner import Runner
from beagle.types import Task, TaskContext, TaskResult


def _bench(outcomes: dict[str, tuple[bool, str | None]], seen: list[str], store: dict):
    """A fake benchmark. ``rollout`` runs per ``outcomes`` (task_id → (resolved, error)),
    appends handled ids to ``seen``, and persists results into ``store[run_dir]`` — the
    stand-in for a harness's native on-disk tree. ``completed`` reads that store back (as
    ``HarborHarness.completed`` reads harbor's result.json), which is the resume seam."""

    class _Harness:
        def rollout(self, agent, items, *, runtime, run_dir, parallelism, retry=None,
                timeout_multiplier=1.0, attempt=0, resuming=False):  # noqa: ANN001
            seen.extend(t.task_id for t, _ in items)
            out = []
            for t, _ in items:
                resolved, error = outcomes.get(t.task_id, (False, None))
                r = TaskResult(
                    task_id=t.task_id, resolved=resolved, error=error,
                    reward=1.0 if resolved else 0.0,
                    tokens={} if error else {"prompt": 10, "completion": 1},
                    artifact_dir=run_dir / "b" / t.task_id)
                store.setdefault(str(run_dir), {})[t.task_id] = r
                out.append(r)
            return out

        def completed(self, items, *, run_dir):  # noqa: ANN001 — returns ALL prior (Runner filters)
            done = store.get(str(run_dir), {})
            return [done[t.task_id] for t, _ in items if t.task_id in done]

    class _Grader:
        def grade(self, results, *, runtime, run_dir, parallelism=1):  # noqa: ANN001
            res = sum(1 for r in results if r.resolved)
            return GradeReport(num_tasks=len(results), num_resolved=res,
                               score=res / len(results) if results else 0.0)

    class _Bench:
        def harness(self):
            return _Harness()

        def grader(self):
            return _Grader()

    return _Bench()


def _dataset(*ids: str):
    return [(Task(task_id=i, benchmark="b"), TaskContext(image="img")) for i in ids]


def _cfg(*ids: str) -> RunConfig:
    return RunConfig.from_dict({
        "model": {"name": "gpt-5.5"}, "agent": {"name": "monet", "config": {}},
        "benchmark": {"name": "b", "task_ids": list(ids)},
    })


def _install(monkeypatch, outcomes) -> list[str]:
    seen: list[str] = []
    store: dict = {}  # persists across runner.run() calls → simulates the native tree on disk
    monkeypatch.setattr(benchmarks, "get", lambda name: _bench(outcomes, seen, store))
    return seen


def test_runner_writes_run_json_and_reduces(tmp_path, monkeypatch) -> None:
    _install(monkeypatch, {"t1": (True, None), "t2": (False, None)})
    rr = Runner(parallelism=2, results_root=tmp_path).run(
        agent=object(), dataset=_dataset("t1", "t2"), config=_cfg("t1", "t2"),
        run_id="RID", config_path="c.yaml",
    )
    assert rr.run_id == "RID" and rr.score == 0.5 and len(rr.results) == 2

    rec = json.loads((tmp_path / "RID" / "run.json").read_text())
    # Thin, benchmark-keyed: per-benchmark score under benchmarks; additive totals; no per_task.
    assert rec["benchmarks"]["b"]["score"] == 0.5 and rec["benchmarks"]["b"]["num_tasks"] == 2
    assert rec["totals"]["num_tasks"] == 2 and rec["totals"]["num_benchmarks"] == 1
    assert rec["totals"]["tokens"]["total"] == 22  # 2 × (10 + 1)
    assert "per_task_results" not in rec           # per-task lives in the harness native trees
    assert rec["config_hash"].startswith("sha256:") and rec["config"]["model"]["name"] == "gpt-5.5"
    assert rec["environment"]["python"]  # provenance captured
    assert not (tmp_path / "RID" / "tasks.jsonl").exists()  # no house ledger


def test_runner_eval_parallelism_drives_grade_fanout(tmp_path, monkeypatch) -> None:
    """``parallelism_eval_patches`` sets the grader's patch-EVAL fan-out, independent of the agent
    (patch-GENERATION) parallelism; ``None`` falls back to ``parallelism``."""
    graded_with: list[int] = []

    def _fake_bench():
        class _H:
            def completed(self, items, *, run_dir):  # noqa: ANN001
                return []

            def rollout(self, agent, items, *, runtime, run_dir, parallelism, retry=None,
                    timeout_multiplier=1.0, attempt=0, resuming=False):  # noqa: ANN001
                return [TaskResult(task_id=t.task_id, resolved=True, reward=1.0,
                                   tokens={"prompt": 1, "completion": 1},
                                   artifact_dir=run_dir / "b" / t.task_id) for t, _ in items]

        class _G:
            def grade(self, results, *, runtime, run_dir, parallelism=1):  # noqa: ANN001
                graded_with.append(parallelism)
                return GradeReport(num_tasks=len(results), num_resolved=len(results), score=1.0)

        class _B:
            def harness(self):
                return _H()

            def grader(self):
                return _G()

        return _B()

    monkeypatch.setattr(benchmarks, "get", lambda name: _fake_bench())

    # eval_parallelism set → grade uses it (64), NOT the generation parallelism (8)
    Runner(parallelism=8, eval_parallelism=64, results_root=tmp_path).run(
        agent=object(), dataset=_dataset("t1"), config=_cfg("t1"), run_id="R1", config_path="c.yaml")
    assert graded_with == [64]

    # eval_parallelism unset → grade falls back to `parallelism`
    graded_with.clear()
    Runner(parallelism=8, results_root=tmp_path).run(
        agent=object(), dataset=_dataset("t2"), config=_cfg("t2"), run_id="R2", config_path="c.yaml")
    assert graded_with == [8]


def test_runner_runs_benchmark_groups_concurrently(tmp_path, monkeypatch) -> None:
    # A mix of two benchmarks with parallelism=2 must roll them out CONCURRENTLY (not
    # tb-then-swe). Proven with a 2-party barrier: each group's rollout blocks until the
    # OTHER group's rollout also arrives — if the Runner serialized groups the first would
    # block forever and the barrier times out (BrokenBarrierError) → test fails.
    import threading

    barrier = threading.Barrier(2, timeout=5)
    both_in_flight: list[bool] = []

    class _H:
        def rollout(self, agent, items, *, runtime, run_dir, parallelism, retry=None,
                timeout_multiplier=1.0, attempt=0, resuming=False):  # noqa: ANN001
            barrier.wait()  # only returns once BOTH groups are here at the same time
            both_in_flight.append(True)
            return [TaskResult(task_id=t.task_id, resolved=True, reward=1.0,
                               tokens={"prompt": 1, "completion": 1}) for t, _ in items]

        def completed(self, items, *, run_dir):  # noqa: ANN001
            return []

    class _G:
        def grade(self, results, *, runtime, run_dir, parallelism=1):  # noqa: ANN001
            return GradeReport(num_tasks=len(results), num_resolved=len(results), score=1.0)

    class _B:
        def harness(self):
            return _H()

        def grader(self):
            return _G()

    monkeypatch.setattr(benchmarks, "get", lambda name: _B())

    dataset = [(Task(task_id="t_tb", benchmark="tb"), TaskContext(image="i")),
               (Task(task_id="t_swe", benchmark="swe"), TaskContext(image="i"))]
    cfg = RunConfig.from_dict({"model": {"name": "gpt-5.5"}, "agent": {"name": "monet", "config": {}},
                               "benchmark": {"name": "tb", "task_ids": ["t_tb"]}, "parallelism": 2})
    rr = Runner(parallelism=2, results_root=tmp_path).run(
        agent=object(), dataset=dataset, config=cfg, run_id="RID")

    assert len(both_in_flight) == 2 and rr.score == 1.0
    rec = json.loads((tmp_path / "RID" / "run.json").read_text())
    assert rec["totals"]["num_benchmarks"] == 2 and rec["totals"]["num_tasks"] == 2
    assert set(rec["benchmarks"]) == {"tb", "swe"}      # both groups recorded


def test_runner_explicit_run_dir_overrides_default(tmp_path, monkeypatch) -> None:
    _install(monkeypatch, {"t1": (True, None)})
    out = tmp_path / "custom" / "gate-out"
    rr = Runner(results_root=tmp_path / "unused").run(
        agent=object(), dataset=_dataset("t1"), config=_cfg("t1"),
        run_id="RID", run_dir=out,
    )
    assert rr.artifact_dir == out
    assert (out / "run.json").exists()                      # written to the explicit dir
    assert not (tmp_path / "unused" / "RID").exists()       # NOT the results_root default


def test_runner_resume_reruns_only_errored(tmp_path, monkeypatch) -> None:
    # First run: t1 resolves, t2 errors. Resume (same config) with retry_errors re-runs
    # ONLY t2; t1 is carried forward. run.json still holds both.
    seen = _install(monkeypatch, {"t1": (True, None), "t2": (False, "boom")})
    runner = Runner(results_root=tmp_path)
    runner.run(agent=object(), dataset=_dataset("t1", "t2"), config=_cfg("t1", "t2"), run_id="RID")
    assert seen == ["t1", "t2"]
    seen.clear()

    runner.run(agent=object(), dataset=_dataset("t1", "t2"), config=_cfg("t1", "t2"),
               run_id="RID", resume=True, retry_errors=True)
    assert seen == ["t2"]  # t1 skipped (done, via harness.completed), t2 retried
    rec = json.loads((tmp_path / "RID" / "run.json").read_text())
    assert rec["totals"]["num_tasks"] == 2                  # both aggregated (t1 carried + t2 rerun)
    assert rec["benchmarks"]["b"]["num_tasks"] == 2


def test_runner_resume_without_retry_skips_all(tmp_path, monkeypatch) -> None:
    seen = _install(monkeypatch, {"t1": (True, None), "t2": (False, None)})
    runner = Runner(results_root=tmp_path)
    runner.run(agent=object(), dataset=_dataset("t1", "t2"), config=_cfg("t1", "t2"), run_id="RID")
    seen.clear()
    runner.run(agent=object(), dataset=_dataset("t1", "t2"), config=_cfg("t1", "t2"),
               run_id="RID", resume=True)
    assert seen == []  # both already done → nothing re-rolled
    rec = json.loads((tmp_path / "RID" / "run.json").read_text())
    assert rec["totals"]["num_tasks"] == 2


def test_runner_retry_unresolved_reruns_all_unresolved(tmp_path, monkeypatch) -> None:
    # retry_unresolved re-runs EVERY resolved=False task — t2 (unresolved, no error) AND t3 (errored) —
    # keeping the resolved t1. A superset of retry_errors (which would rerun only t3).
    seen = _install(monkeypatch, {"t1": (True, None), "t2": (False, None), "t3": (False, "boom")})
    runner = Runner(results_root=tmp_path)
    runner.run(agent=object(), dataset=_dataset("t1", "t2", "t3"),
               config=_cfg("t1", "t2", "t3"), run_id="RID")
    seen.clear()
    runner.run(agent=object(), dataset=_dataset("t1", "t2", "t3"), config=_cfg("t1", "t2", "t3"),
               run_id="RID", resume=True, retry_unresolved=True)
    assert seen == ["t2", "t3"]   # both unresolved rerun; the resolved t1 is kept


def test_runner_retry_unresolved_fails_loud_without_signal(tmp_path, monkeypatch) -> None:
    # A harness whose prior trials have NEITHER a reward NOR an error → no resolved signal to key off.
    # retry_unresolved must refuse rather than silently treat them all as "unresolved" and re-run.
    def _blind_bench():
        def _mk(items, run_dir):
            return [TaskResult(task_id=t.task_id, resolved=False, reward=None, error=None,
                               artifact_dir=run_dir / "b" / t.task_id) for t, _ in items]

        class _H:
            def rollout(self, agent, items, *, runtime, run_dir, parallelism, retry=None,
                    timeout_multiplier=1.0, attempt=0, resuming=False):  # noqa: ANN001
                return _mk(items, run_dir)

            def completed(self, items, *, run_dir):  # noqa: ANN001
                return _mk(items, run_dir)

        class _G:
            def grade(self, results, *, runtime, run_dir, parallelism=1):  # noqa: ANN001
                return GradeReport(num_tasks=len(results), num_resolved=0, score=0.0)

        class _B:
            def harness(self):
                return _H()

            def grader(self):
                return _G()

        return _B()

    monkeypatch.setattr(benchmarks, "get", lambda name: _blind_bench())
    with pytest.raises(RuntimeError, match="resolved signal"):
        Runner(results_root=tmp_path).run(
            agent=object(), dataset=_dataset("t1"), config=_cfg("t1"),
            run_id="RID", resume=True, retry_unresolved=True, force_resume=True)


def test_runner_resume_refuses_on_config_drift(tmp_path, monkeypatch) -> None:
    _install(monkeypatch, {"t1": (True, None)})
    runner = Runner(results_root=tmp_path)
    runner.run(agent=object(), dataset=_dataset("t1"), config=_cfg("t1"), run_id="RID")
    with pytest.raises(RuntimeError, match="config changed"):
        runner.run(agent=object(), dataset=_dataset("t1", "t2"), config=_cfg("t1", "t2"),
                   run_id="RID", resume=True)  # task_ids differ → different hash → drift


def test_runner_force_resume_overrides_drift_and_records_both_hashes(tmp_path, monkeypatch) -> None:
    # --force-resume bypasses the drift guard (keeps finished tasks) and records BOTH the prior and
    # current config_hash in run.json (config_hash_drift), so the mixed-config run stays auditable.
    import json

    _install(monkeypatch, {"t1": (True, None), "t2": (True, None)})
    runner = Runner(results_root=tmp_path)
    first = runner.run(agent=object(), dataset=_dataset("t1"), config=_cfg("t1"), run_id="RID")
    prior_hash = json.loads((tmp_path / "RID" / "run.json").read_text())["config_hash"]

    # A drifted config (extra task → new hash) resumes cleanly with force_resume=True.
    rr = runner.run(agent=object(), dataset=_dataset("t1", "t2"), config=_cfg("t1", "t2"),
                    run_id="RID", resume=True, force_resume=True)
    rec = json.loads((tmp_path / "RID" / "run.json").read_text())
    assert rec["config_hash_drift"] == prior_hash          # prior hash recorded
    assert rec["config_hash"] != prior_hash                # current (drifted) hash is the run's
    assert rr.metrics["num_tasks"] == 2                    # t1 (resumed) + t2 (new) both present
    assert first.run_id == "RID"


def _recording_bench(record: list, *, old_signature: bool = False):
    """A benchmark whose harness() records the env_import_path it's called with (or, with
    ``old_signature``, keeps the pre-#20 ``harness(self)`` shape to prove backward-compat)."""

    class _H:
        def completed(self, items, *, run_dir):  # noqa: ANN001
            return []

        def rollout(self, agent, items, *, runtime, run_dir, parallelism, retry=None,
                timeout_multiplier=1.0, attempt=0, resuming=False):  # noqa: ANN001
            return [TaskResult(task_id=t.task_id, resolved=True, reward=1.0,
                               tokens={"prompt": 1, "completion": 1},
                               artifact_dir=run_dir / "b" / t.task_id) for t, _ in items]

    class _G:
        def grade(self, results, *, runtime, run_dir, parallelism=1):  # noqa: ANN001
            return GradeReport(num_tasks=len(results), num_resolved=len(results), score=1.0)

    class _B:
        if old_signature:
            def harness(self):
                record.append("PLAIN")
                return _H()
        else:
            def harness(self, env_import_path=None):
                record.append(env_import_path)
                return _H()

        def grader(self):
            return _G()

    return _B()


def test_runner_threads_env_import_path_from_benchmark_options(tmp_path, monkeypatch) -> None:
    # #20 item 1: benchmark.options.env_import_path reaches Benchmark.harness() through the runner.
    rec: list = []
    monkeypatch.setattr(benchmarks, "get", lambda name: _recording_bench(rec))
    cfg = RunConfig.from_dict({
        "model": {"name": "gpt-5.5"}, "agent": {"name": "monet", "config": {}},
        "benchmark": {"name": "b", "task_ids": ["t1"],
                      "options": {"env_import_path": "my.mod:LocalEnv"}}})
    Runner(results_root=tmp_path).run(agent=object(), dataset=_dataset("t1"), config=cfg, run_id="RID")
    assert rec == ["my.mod:LocalEnv"]                       # the option flowed through


def test_runner_calls_plain_harness_when_env_import_path_unset(tmp_path, monkeypatch) -> None:
    # Backward-compat: no option → runner calls the plain harness() (a benchmark with the old
    # signature must not break).
    rec: list = []
    monkeypatch.setattr(benchmarks, "get", lambda name: _recording_bench(rec, old_signature=True))
    Runner(results_root=tmp_path).run(agent=object(), dataset=_dataset("t1"), config=_cfg("t1"), run_id="RID")
    assert rec == ["PLAIN"]                                 # plain harness() invoked, no TypeError
