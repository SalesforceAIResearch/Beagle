"""Unit tests for the shared content-retry dir consolidator.

``consolidate_retry_dirs`` folds each ``<job_id>-retry<N>`` sibling back into the
attempt-0 ``<job_id>`` dir so a sweep leaves ONE trial per task and no stray retry
folders. The winner is the LATEST round that ran a task (a task stops being retried
once it passes, so its highest round is the passing trial when one exists).
"""

from __future__ import annotations

import json
from pathlib import Path

from xrlenv_plugins.benchmarks._sweep_retry import (
    consolidate_retry_dirs,
    consolidate_swebench_eval_dirs,
)


def _trial(job_dir: Path, task_id: str, suffix: str, *, with_result: bool = True) -> Path:
    """Create a trial dir ``<job_dir>/<task_id>__<suffix>`` with a result.json whose
    ``config.task.path`` basename is the canonical task id (as harbor/pier write it)."""
    d = job_dir / f"{task_id}__{suffix}"
    d.mkdir(parents=True)
    if with_result:
        (d / "result.json").write_text(json.dumps(
            {"config": {"task": {"path": f"/cache/bench/{task_id}"}}}
        ))
    return d


def _task_ids(job_dir: Path) -> set[str]:
    """Canonical task ids present in a job dir (basename of config.task.path)."""
    out: set[str] = set()
    for d in job_dir.iterdir():
        if d.is_dir():
            j = json.loads((d / "result.json").read_text())
            out.add(Path(j["config"]["task"]["path"]).name)
    return out


def test_no_retry_dirs_is_noop(tmp_path: Path) -> None:
    main = tmp_path / "bench-full-TS"
    _trial(main, "taskA", "aaa")
    folded = consolidate_retry_dirs(tmp_path, "bench-full-TS", 2)
    assert folded == []
    assert {p.name for p in main.iterdir()} == {"taskA__aaa"}


def test_retry_pass_replaces_attempt0_failure(tmp_path: Path) -> None:
    jid = "bench-full-TS"
    main = tmp_path / jid
    _trial(main, "taskA", "aaa")          # passed attempt 0 — never retried
    _trial(main, "taskB", "bbb")          # failed attempt 0
    r1 = tmp_path / f"{jid}-retry1"
    _trial(r1, "taskB", "ccc")            # passed on retry 1

    folded = consolidate_retry_dirs(tmp_path, jid, 2)

    assert folded == [f"{jid}-retry1"]
    assert not r1.exists()                                  # retry dir removed
    names = {p.name for p in main.iterdir()}
    assert names == {"taskA__aaa", "taskB__ccc"}            # bbb superseded by ccc
    assert _task_ids(main) == {"taskA", "taskB"}            # still one per task


def test_latest_round_wins_across_two_retries(tmp_path: Path) -> None:
    jid = "bench-full-TS"
    main = tmp_path / jid
    _trial(main, "taskB", "bbb")          # attempt 0 fail
    _trial(tmp_path / f"{jid}-retry1", "taskB", "ccc")   # retry 1 fail
    _trial(tmp_path / f"{jid}-retry2", "taskB", "ddd")   # retry 2 (winner)

    folded = consolidate_retry_dirs(tmp_path, jid, 2)

    assert folded == [f"{jid}-retry1", f"{jid}-retry2"]
    assert not (tmp_path / f"{jid}-retry1").exists()
    assert not (tmp_path / f"{jid}-retry2").exists()
    assert {p.name for p in main.iterdir()} == {"taskB__ddd"}   # only the latest


def test_untouched_tasks_survive(tmp_path: Path) -> None:
    jid = "bench-full-TS"
    main = tmp_path / jid
    _trial(main, "keepme", "k1")          # never retried
    _trial(main, "flaky", "f0")
    _trial(tmp_path / f"{jid}-retry1", "flaky", "f1")

    consolidate_retry_dirs(tmp_path, jid, 2)

    assert _task_ids(main) == {"keepme", "flaky"}
    assert (main / "keepme__k1").exists()             # untouched
    assert (main / "flaky__f1").exists()
    assert not (main / "flaky__f0").exists()


def test_converged_early_missing_higher_round(tmp_path: Path) -> None:
    """content_retries=2 but the sweep converged at round 1, so no retry2 dir
    exists — the absent round is skipped, not an error."""
    jid = "bench-full-TS"
    main = tmp_path / jid
    _trial(main, "taskB", "bbb")
    _trial(tmp_path / f"{jid}-retry1", "taskB", "ccc")

    folded = consolidate_retry_dirs(tmp_path, jid, 2)

    assert folded == [f"{jid}-retry1"]           # only round 1 folded; round 2 absent
    assert {p.name for p in main.iterdir()} == {"taskB__ccc"}


def test_content_retries_bound_is_respected(tmp_path: Path) -> None:
    """A stray higher-numbered retry dir beyond content_retries is left alone."""
    jid = "bench-full-TS"
    main = tmp_path / jid
    _trial(main, "taskB", "bbb")
    _trial(tmp_path / f"{jid}-retry1", "taskB", "ccc")
    stray = tmp_path / f"{jid}-retry2"
    _trial(stray, "taskB", "ddd")

    folded = consolidate_retry_dirs(tmp_path, jid, 1)   # only look at round 1

    assert folded == [f"{jid}-retry1"]
    assert stray.exists()                               # round 2 untouched
    assert {p.name for p in main.iterdir()} == {"taskB__ccc"}


def test_fallback_task_id_from_dir_name(tmp_path: Path) -> None:
    """When result.json is missing/unreadable, the task id falls back to the
    dir name with its __<suffix> stripped."""
    jid = "bench-full-TS"
    main = tmp_path / jid
    _trial(main, "taskB", "bbb", with_result=False)
    _trial(tmp_path / f"{jid}-retry1", "taskB", "ccc", with_result=False)

    consolidate_retry_dirs(tmp_path, jid, 2)

    assert {p.name for p in main.iterdir()} == {"taskB__ccc"}


def test_missing_main_dir_is_safe(tmp_path: Path) -> None:
    """No attempt-0 dir → nothing to consolidate into; return [] without raising."""
    _trial(tmp_path / "bench-full-TS-retry1", "taskB", "ccc")
    folded = consolidate_retry_dirs(tmp_path, "bench-full-TS", 2)
    assert folded == []
    assert (tmp_path / "bench-full-TS-retry1").exists()   # left in place, untouched


def test_non_dir_entries_in_retry_are_ignored(tmp_path: Path) -> None:
    """A stray file (e.g. a job-level summary) in a retry dir is skipped, not moved."""
    jid = "bench-full-TS"
    main = tmp_path / jid
    _trial(main, "taskB", "bbb")
    r1 = tmp_path / f"{jid}-retry1"
    _trial(r1, "taskB", "ccc")
    (r1 / "job-summary.json").write_text("{}")

    consolidate_retry_dirs(tmp_path, jid, 2)

    assert not r1.exists()                                # whole retry dir removed
    assert {p.name for p in main.iterdir()} == {"taskB__ccc"}


# ── consolidate_swebench_eval_dirs (SWE run_evaluation folders) ────────────────


def _swe_inst(eval_root: Path, run_id: str, inst: str,
              model: str = "xrlenv-oracle") -> None:
    """A swebench eval dir <eval_root>/<run_id>/<model>/<inst>/run_instance.log; the
    log content is the run_id so tests can assert which attempt's log was kept."""
    d = eval_root / run_id / model / inst
    d.mkdir(parents=True)
    (d / "run_instance.log").write_text(run_id)


def test_swebench_consolidate_folds_retry_folder(tmp_path: Path) -> None:
    er = tmp_path / "logs" / "run_evaluation"
    _swe_inst(er, "xrlenv-oracle-sweep", "a-1")
    _swe_inst(er, "xrlenv-oracle-sweep", "b-2")
    _swe_inst(er, "xrlenv-oracle-sweep", "c-3")            # superseded by the retry
    _swe_inst(er, "xrlenv-oracle-sweep-retry1", "c-3")     # latest for c-3

    folded = consolidate_swebench_eval_dirs(tmp_path, "xrlenv-oracle-sweep")

    assert folded == ["xrlenv-oracle-sweep-retry1"]
    assert not (er / "xrlenv-oracle-sweep-retry1").exists()
    main = er / "xrlenv-oracle-sweep" / "xrlenv-oracle"
    assert {p.name for p in main.iterdir()} == {"a-1", "b-2", "c-3"}
    assert (main / "c-3" / "run_instance.log").read_text() == "xrlenv-oracle-sweep-retry1"


def test_swebench_consolidate_latest_attempt_wins(tmp_path: Path) -> None:
    # sorted name order -infra1 < -retry1 < -retry2 → retry2 (the latest) wins.
    er = tmp_path / "logs" / "run_evaluation"
    for rid in ("xrlenv-oracle-sweep", "xrlenv-oracle-sweep-infra1",
                "xrlenv-oracle-sweep-retry1", "xrlenv-oracle-sweep-retry2"):
        _swe_inst(er, rid, "x-1")

    consolidate_swebench_eval_dirs(tmp_path, "xrlenv-oracle-sweep")

    assert [d.name for d in er.iterdir()] == ["xrlenv-oracle-sweep"]   # extras gone
    main = er / "xrlenv-oracle-sweep" / "xrlenv-oracle"
    assert (main / "x-1" / "run_instance.log").read_text() == "xrlenv-oracle-sweep-retry2"


def test_swebench_consolidate_noop_without_retries(tmp_path: Path) -> None:
    er = tmp_path / "logs" / "run_evaluation"
    _swe_inst(er, "xrlenv-oracle-sweep", "a-1")
    assert consolidate_swebench_eval_dirs(tmp_path, "xrlenv-oracle-sweep") == []
    assert (er / "xrlenv-oracle-sweep" / "xrlenv-oracle" / "a-1").exists()


def test_swebench_consolidate_missing_root_is_safe(tmp_path: Path) -> None:
    assert consolidate_swebench_eval_dirs(tmp_path, "xrlenv-oracle-sweep") == []
