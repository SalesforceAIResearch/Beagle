"""Hermetic tests for :class:`SweBenchGrader.evaluate_patch`.

The real swebench evaluator spins docker containers and cannot run in CI, so we
monkeypatch ``swebench.harness.run_evaluation.main`` to a no-op that (optionally)
writes a fake per-instance ``report.json`` in the exact native layout the grader
reads, then assert the resolved verdict the grader derives from it.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from beagle.benchmarks.swe_bench_verified import SweBenchGrader
from beagle.types import RolloutStatus, TaskResult


class _FakeRuntime:
    """Stand-in for ``ContainerRuntime`` — the grader accepts it but the swebench
    evaluator drives its own containers, so nothing is called on it."""


def _fake_main_factory(*, resolved: bool | None, calls: list) -> object:
    """Build a fake ``run_evaluation.main`` that records its call and, unless
    ``resolved is None`` (the "missing report" case), writes a native-layout
    ``report.json`` under the current working directory.

    The grader chdir's into its temp run dir before calling ``main`` and passes
    ``report_dir="."``, so writing relative to ``Path.cwd()`` reproduces the real
    ``logs/run_evaluation/<run_id>/<model>/<instance>/report.json`` layout.
    """

    def _fake_main(**kwargs) -> None:
        calls.append(kwargs)
        if resolved is None:
            return  # simulate the harness leaving no report (unscored tail)
        run_id = kwargs["run_id"]
        instance_id = kwargs["instance_ids"][0]
        report_dir = (
            Path.cwd()
            / "logs"
            / "run_evaluation"
            / run_id
            / "beagle"
            / instance_id
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.json").write_text(
            json.dumps({instance_id: {"resolved": resolved}}),
            encoding="utf-8",
        )

    return _fake_main


def _install_fake_swebench(monkeypatch: pytest.MonkeyPatch, fake_main: object) -> None:
    """Inject a fake ``swebench.harness.run_evaluation`` module so the grader's
    lazy ``from swebench.harness.run_evaluation import main`` resolves to our stub
    — no real swebench import."""
    run_eval_mod = types.ModuleType("swebench.harness.run_evaluation")
    run_eval_mod.main = fake_main  # type: ignore[attr-defined]
    harness_mod = types.ModuleType("swebench.harness")
    root_mod = types.ModuleType("swebench")
    monkeypatch.setitem(sys.modules, "swebench", root_mod)
    monkeypatch.setitem(sys.modules, "swebench.harness", harness_mod)
    monkeypatch.setitem(sys.modules, "swebench.harness.run_evaluation", run_eval_mod)


def test_resolved_true_scores_1(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    _install_fake_swebench(monkeypatch, _fake_main_factory(resolved=True, calls=calls))
    grader = SweBenchGrader()
    score = grader.evaluate_patch("django__django-1", "diff --git a b", runtime=_FakeRuntime())
    assert score == 1.0
    assert len(calls) == 1
    # It graded exactly this instance with the single-entry predictions schema.
    assert calls[0]["instance_ids"] == ["django__django-1"]
    assert calls[0]["dataset_name"] == "SWE-bench/SWE-Bench_Verified"


def test_resolved_false_scores_0(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    _install_fake_swebench(monkeypatch, _fake_main_factory(resolved=False, calls=calls))
    grader = SweBenchGrader()
    score = grader.evaluate_patch("django__django-2", "diff --git a b", runtime=_FakeRuntime())
    assert score == 0.0
    assert len(calls) == 1


def test_missing_report_scores_0(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    # resolved=None → the fake main writes no report (unscored-tail simulation).
    _install_fake_swebench(monkeypatch, _fake_main_factory(resolved=None, calls=calls))
    grader = SweBenchGrader()
    score = grader.evaluate_patch("django__django-3", "diff --git a b", runtime=_FakeRuntime())
    assert score == 0.0
    assert len(calls) == 1  # swebench WAS invoked; it just left no report


def test_empty_patch_skips_swebench(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    _install_fake_swebench(monkeypatch, _fake_main_factory(resolved=True, calls=calls))
    grader = SweBenchGrader()
    assert grader.evaluate_patch("django__django-4", "", runtime=_FakeRuntime()) == 0.0
    assert grader.evaluate_patch("django__django-4", "   \n ", runtime=_FakeRuntime()) == 0.0
    # swebench must NOT be invoked for an empty/whitespace patch.
    assert calls == []


def test_grade_batches_all_patches_in_one_native_run(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # The core parallelism fix: grade() assembles ONE predictions.json and calls run_evaluation
    # ONCE with max_workers=parallelism (swebench's native concurrency), not a serial per-task loop.
    calls: list = []
    resolved = {"aa__aa-1": True, "bb__bb-2": False}

    def fake_main(**kwargs) -> None:
        calls.append(kwargs)
        for iid in kwargs["instance_ids"]:  # batch: write a native report per instance
            d = Path.cwd() / "logs" / "run_evaluation" / kwargs["run_id"] / "beagle" / iid
            d.mkdir(parents=True, exist_ok=True)
            (d / "report.json").write_text(json.dumps({iid: {"resolved": resolved[iid]}}))
            (d / "run_instance.log").write_text(f"log {iid}")
            (d / "patch.diff").write_text(f"patch {iid}")

    _install_fake_swebench(monkeypatch, fake_main)
    bench = "swe-bench-verified"
    results = []
    for iid, patch in [("aa__aa-1", "diff a"), ("bb__bb-2", "diff b"), ("cc__cc-3", "")]:
        ad = tmp_path / bench / iid
        ad.mkdir(parents=True, exist_ok=True)  # the task dir the base rollout would have made
        results.append(TaskResult(task_id=iid, benchmark=bench, patch=patch or None,
                                  status=RolloutStatus.COMPLETED, artifact_dir=ad))

    report = SweBenchGrader().grade(results, runtime=_FakeRuntime(), run_dir=tmp_path, parallelism=3)

    # ONE batch invocation; the two non-empty patches; native max_workers = parallelism.
    assert len(calls) == 1
    assert sorted(calls[0]["instance_ids"]) == ["aa__aa-1", "bb__bb-2"]   # empty patch excluded
    assert calls[0]["max_workers"] == 3
    # unified predictions.json (dict keyed by instance_id — swebench's native schema)
    preds = json.loads((tmp_path / bench / "predictions.json").read_text())
    assert set(preds) == {"aa__aa-1", "bb__bb-2"} and preds["aa__aa-1"]["model_patch"] == "diff a"
    # rewards/resolved applied; empty patch scored 0 without an evaluator container
    by = {r.task_id: r for r in results}
    assert by["aa__aa-1"].reward == 1.0 and by["aa__aa-1"].resolved
    assert by["bb__bb-2"].reward == 0.0 and not by["bb__bb-2"].resolved
    # empty patch → scored 0 AND flagged a retryable NoAttempt error (so --retry-errors catches it
    # instead of it reading as a genuine reward=0 capability failure)
    assert by["cc__cc-3"].reward == 0.0
    assert (by["cc__cc-3"].error or "").startswith("NoAttempt")
    # native per-instance files distributed into the co-located task dir + result.json written
    assert (tmp_path / bench / "aa__aa-1" / "run_instance.log").read_text() == "log aa__aa-1"
    rj = json.loads((tmp_path / bench / "aa__aa-1" / "result.json").read_text())
    assert rj["resolved"] is True and rj["reward"] == 1.0
    assert report.num_tasks == 3 and report.num_resolved == 1 and report.score == pytest.approx(1 / 3)
    # canonical benchmark-level result.json (fixed name, unlike swebench's <model>.<run_id>.json) —
    # one stable path across benchmark families for downstream parsing.
    brj = json.loads((tmp_path / bench / "result.json").read_text())
    assert brj["benchmark"] == bench and brj["num_tasks"] == 3 and brj["num_resolved"] == 1
    assert brj["num_errored"] == 1 and brj["score"] == pytest.approx(1 / 3)  # cc: empty patch → NoAttempt
    assert brj["per_task"]["aa__aa-1"] == {"resolved": True, "reward": 1.0}
    assert brj["per_task"]["bb__bb-2"] == {"resolved": False, "reward": 0.0}


def test_grade_preserves_resumed_reward_and_skips_reeval(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """On resume, an already-graded task returns with its reward set but no patch (result.json doesn't
    store the patch). grade() must PRESERVE that reward (not re-score it to 0) and NOT re-run swebench
    for it — only the new patch-bearing task is evaluated."""
    calls: list = []

    def fake_main(**kwargs) -> None:
        calls.append(kwargs)
        for iid in kwargs["instance_ids"]:
            d = Path.cwd() / "logs" / "run_evaluation" / kwargs["run_id"] / "beagle" / iid
            d.mkdir(parents=True, exist_ok=True)
            (d / "report.json").write_text(json.dumps({iid: {"resolved": True}}))

    _install_fake_swebench(monkeypatch, fake_main)
    bench = "swe-bench-verified"
    (tmp_path / bench / "done__1").mkdir(parents=True)
    (tmp_path / bench / "new__2").mkdir(parents=True)
    done = TaskResult(task_id="done__1", benchmark=bench, status=RolloutStatus.COMPLETED,
                      reward=1.0, resolved=True, patch=None,          # resumed: graded before, no patch
                      artifact_dir=tmp_path / bench / "done__1")
    new = TaskResult(task_id="new__2", benchmark=bench, status=RolloutStatus.COMPLETED,
                     patch="diff new", artifact_dir=tmp_path / bench / "new__2")

    report = SweBenchGrader().grade([done, new], runtime=_FakeRuntime(), run_dir=tmp_path, parallelism=2)

    # swebench ran ONLY for the new patch-bearing task — the resumed one was NOT re-evaluated
    assert len(calls) == 1 and calls[0]["instance_ids"] == ["new__2"]
    # resumed reward preserved (would have been clobbered to 0 before the fix); new task graded
    assert done.reward == 1.0 and done.resolved is True
    assert new.reward == 1.0 and new.resolved is True
    assert report.num_tasks == 2 and report.num_resolved == 2


def test_cluster_env_installs_docker_drop_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """With XRLENV_GRPC_HOST set, the grader swaps ``docker.from_env`` for xrlenv's
    (before importing swebench). With it unset, it does not."""
    calls: list = []
    _install_fake_swebench(monkeypatch, _fake_main_factory(resolved=True, calls=calls))

    # Fake docker + xrlenv.compat.docker_client so the drop-in has something to swap.
    sentinel_original = object()
    fake_docker = types.ModuleType("docker")
    fake_docker.from_env = sentinel_original  # type: ignore[attr-defined]

    def _xrlenv_from_env(*a, **k):  # pragma: no cover - identity sentinel only
        return None

    docker_client_mod = types.ModuleType("xrlenv.compat.docker_client")
    docker_client_mod.from_env = _xrlenv_from_env  # type: ignore[attr-defined]
    compat_mod = types.ModuleType("xrlenv.compat")
    xrlenv_root = types.ModuleType("xrlenv")
    monkeypatch.setitem(sys.modules, "docker", fake_docker)
    monkeypatch.setitem(sys.modules, "xrlenv", xrlenv_root)
    monkeypatch.setitem(sys.modules, "xrlenv.compat", compat_mod)
    monkeypatch.setitem(sys.modules, "xrlenv.compat.docker_client", docker_client_mod)

    # Cluster: drop-in installed → docker.from_env replaced with xrlenv's.
    monkeypatch.setenv("XRLENV_GRPC_HOST", "grpc.example.invalid")
    SweBenchGrader().evaluate_patch("django__django-5", "diff", runtime=_FakeRuntime())
    assert fake_docker.from_env is _xrlenv_from_env

    # Local: reset the sentinel, unset the host → drop-in NOT installed.
    fake_docker.from_env = sentinel_original  # type: ignore[attr-defined]
    monkeypatch.delenv("XRLENV_GRPC_HOST", raising=False)
    SweBenchGrader().evaluate_patch("django__django-6", "diff", runtime=_FakeRuntime())
    assert fake_docker.from_env is sentinel_original
