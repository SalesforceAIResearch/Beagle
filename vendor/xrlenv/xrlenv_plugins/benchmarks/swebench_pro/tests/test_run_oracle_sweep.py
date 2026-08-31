from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from xrlenv_plugins.benchmarks.swebench_pro import run_oracle_sweep as ros


def _tr(task: str, rewards, exc=None):
    return SimpleNamespace(config=SimpleNamespace(task=SimpleNamespace(path=Path("/cache/swebench-pro") / task)),
                           task_name="swebench-pro/" + task, verifier_result=SimpleNamespace(rewards=rewards) if rewards is not None else None,
                           exception_info=SimpleNamespace(exception_type=exc) if exc else None)


def test_pass_gate_and_task_key():
    assert ros._trial_passes(_tr("t", {"reward": 1.0})) == (True, None)
    assert ros._trial_passes(_tr("t", {"reward": 1, "resolved": 1, "p2p_total": 0, "p2p_passed": 0})) == (True, None)   # diagnostics may be 0
    assert ros._trial_passes(_tr("t", {"f2p": 1.0}))[1] == "no verifier reward recorded"
    assert ros._trial_passes(_tr("t", {"reward": 0.0}))[0] is False
    assert ros._trial_passes(_tr("t", None))[0] is False
    assert ros._trial_passes(_tr("t", {"reward": 1.0}, exc="VerifierTimeoutError"))[1].startswith("exception")
    assert ros._task_key(_tr("instance_x", {"reward": 1})) == "instance_x"


def test_resolve_tasks_from_list_and_file(tmp_path: Path):
    shard = tmp_path / "swebench-pro"
    for t in ("instance_a", "instance_b"):
        (shard / t / "solution").mkdir(parents=True)
        (shard / t / "solution" / "solve.sh").write_text("#!/bin/bash\n")
    assert ros._resolve_tasks(shard, None) == ["instance_a", "instance_b"]
    assert ros._resolve_tasks(shard, "instance_b") == ["instance_b"]
    f = tmp_path / "ids"
    f.write_text("instance_a\n# c\ninstance_b\n")
    assert ros._resolve_tasks(shard, str(f)) == ["instance_a", "instance_b"]
    with pytest.raises(SystemExit):
        ros._resolve_tasks(shard, "instance_zzz")
    with pytest.raises(SystemExit):
        ros._resolve_tasks(shard, "")


def test_job_config_retries_are_infra_only(tmp_path: Path):
    pytest.importorskip("harbor")
    cfg = ros._build_job_config(task_ids=["instance_a"], dataset_root=tmp_path, jobs_dir=tmp_path, job_id="j", n_concurrent_trials=2, retries=3)
    assert cfg.retry.max_retries == 3 and set(cfg.retry.include_exceptions) == set(ros._INFRA_RETRY_EXCEPTIONS)
    assert "VerifierTimeoutError" not in cfg.retry.include_exceptions and cfg.environment.import_path == ros.ENV_IMPORT_PATH
    assert cfg.environment.kwargs.get("xrlenv_cpu_pinning") is True and cfg.tasks[0].path == tmp_path / "instance_a"


def test_long_comma_list_never_stats(tmp_path: Path):
    ids = [f"instance_repo__repo-{'a' * 40}-v{'b' * 40}-{i}" for i in range(6)]
    assert ros._ids_from_arg(",".join(ids)) == ids                       # > NAME_MAX joined; must not raise OSError
    f = tmp_path / "ids.txt"
    f.write_text("\n".join(ids[:2]) + "\n")
    assert ros._ids_from_arg(str(f)) == ids[:2]
