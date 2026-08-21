"""Unit tests for the terminal-bench-2-1 build-plan generator.

The load-bearing, deterministic piece is that each entry's ``image_ref`` comes
from the task's authoritative ``[environment] docker_image`` — so mixed upstream
tags (some tasks on ``:20251031``, some rebuilt at ``:20260403`` / ``:20260430``)
survive into the plan verbatim rather than being flattened to one hard-coded tag.
Offline: a synthetic cache shard in tmp, no FSx, no Docker Hub (``probe_sizes``
is off so the generator never touches the network).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from xrlenv_plugins.benchmarks.terminal_bench_2_1 import build_plan_gen as gen


def _write_task(shard: Path, task_id: str, docker_image: str | None) -> None:
    """Materialize a minimal task dir: ``solution/solve.sh`` (the discovery
    predicate) + a ``task.toml`` with (optionally) an ``[environment]
    docker_image``."""
    (shard / task_id / "solution").mkdir(parents=True)
    (shard / task_id / "solution" / "solve.sh").write_text("#!/bin/bash\n")
    img = f'docker_image = "{docker_image}"\n' if docker_image else ""
    (shard / task_id / "task.toml").write_text(
        "[environment]\n" + img + 'cpus = 2\n',
    )


def test_generate_plan_reads_per_task_docker_image(tmp_path: Path) -> None:
    shard = tmp_path / gen.SHARD
    _write_task(shard, "old-tag-task", "alexgshaw/old-tag-task:20251031")
    _write_task(shard, "new-tag-task", "alexgshaw/new-tag-task:20260430")

    plan = gen.generate_plan(
        ["old-tag-task", "new-tag-task"], shard_dir=shard, probe_sizes=False,
    )

    assert plan["version"] == 1
    assert plan["name"] == "terminal-bench-2-1-2-task"
    refs = [e["image_ref"] for e in plan["entries"]]
    # Each ref is the task's own docker_image — the two distinct tags both survive.
    assert refs == [
        "alexgshaw/old-tag-task:20251031",
        "alexgshaw/new-tag-task:20260430",
    ]
    for entry in plan["entries"]:
        assert entry["context_source"] == {"type": "registry"}
        assert entry["placement"]["size_hint_source"] == "heuristic"
        assert entry["placement"]["size_hint_bytes"] == gen.DEFAULT_SIZE_HINT_BYTES
        assert entry["pinned"] is False
        assert entry["priority"] == 0


def test_pinned_and_home_count_flow_through(tmp_path: Path) -> None:
    shard = tmp_path / gen.SHARD
    _write_task(shard, "t", "alexgshaw/t:20251031")
    plan = gen.generate_plan(
        ["t"], shard_dir=shard, probe_sizes=False,
        pinned=True, preferred_home_count=3,
    )
    entry = plan["entries"][0]
    assert entry["pinned"] is True
    assert entry["placement"]["preferred_home_count"] == 3


def test_missing_docker_image_fails_loud(tmp_path: Path) -> None:
    shard = tmp_path / gen.SHARD
    _write_task(shard, "no-image", docker_image=None)
    with pytest.raises(SystemExit, match=r"no \[environment\] docker_image"):
        gen.generate_plan(["no-image"], shard_dir=shard, probe_sizes=False)


def test_missing_task_fails_loud(tmp_path: Path) -> None:
    shard = tmp_path / gen.SHARD
    shard.mkdir(parents=True)
    with pytest.raises(SystemExit, match="is the shard populated"):
        gen.generate_plan(["ghost"], shard_dir=shard, probe_sizes=False)


def test_discover_all_tasks(tmp_path: Path) -> None:
    shard = tmp_path / gen.SHARD
    _write_task(shard, "b-task", "alexgshaw/b-task:20251031")
    _write_task(shard, "a-task", "alexgshaw/a-task:20251031")
    # A dir without solution/solve.sh is not a task and must be skipped.
    (shard / "not-a-task").mkdir()
    assert gen._discover_all_tasks(shard) == ["a-task", "b-task"]
    assert gen._discover_all_tasks(tmp_path / "absent") == []


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("alexgshaw/fix-git:20260403", ("alexgshaw/fix-git", "20260403")),
        ("alexgshaw/dna-insert:20251031", ("alexgshaw/dna-insert", "20251031")),
        ("alexgshaw/no-tag", ("alexgshaw/no-tag", "latest")),
        # A registry host:port prefix must not be mistaken for the tag.
        ("host:5011/ns/name", ("host:5011/ns/name", "latest")),
        ("host:5011/ns/name:main", ("host:5011/ns/name", "main")),
    ],
)
def test_split_repo_tag(ref: str, expected: tuple[str, str]) -> None:
    assert gen._split_repo_tag(ref) == expected
