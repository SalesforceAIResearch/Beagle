"""Unit tests for the DeepSWE build-plan generator (pure logic; no network)."""
from __future__ import annotations

from pathlib import Path

import pytest
from xrlenv_plugins.benchmarks.deep_swe.build_plan_gen import (
    DEFAULT_SIZE_HINT_BYTES,
    _discover_all_tasks,
    _split_repo_tag,
    _task_image_ref,
    generate_plan,
)


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        # public ECR host:port-free ref with a tag
        (
            "public.ecr.aws/d3j8x8q7/swe-bench-202605:kh-v1.1",
            ("public.ecr.aws/d3j8x8q7/swe-bench-202605", "kh-v1.1"),
        ),
        # registry host:PORT prefix must not be mistaken for the tag
        ("reg:5011/deep-swe/t", ("reg:5011/deep-swe/t", "latest")),
        # plain, no tag
        ("busybox", ("busybox", "latest")),
    ],
)
def test_split_repo_tag(ref: str, expected: tuple[str, str]) -> None:
    assert _split_repo_tag(ref) == expected


def _make_task(shard: Path, name: str, *, docker_image: str | None) -> None:
    d = shard / name
    d.mkdir(parents=True)
    body = "[environment]\n"
    if docker_image is not None:
        body += f'docker_image = "{docker_image}"\n'
    (d / "task.toml").write_text(body)


def test_discover_and_task_image_ref(tmp_path: Path) -> None:
    shard = tmp_path / "deep-swe"
    _make_task(shard, "b-task", docker_image="public.ecr.aws/x/img:b")
    _make_task(shard, "a-task", docker_image="public.ecr.aws/x/img:a")
    (shard / "stray").mkdir()  # no task.toml -> ignored

    assert _discover_all_tasks(shard) == ["a-task", "b-task"]  # sorted
    assert _task_image_ref(shard, "a-task") == "public.ecr.aws/x/img:a"


def test_task_image_ref_fails_loud_without_docker_image(tmp_path: Path) -> None:
    shard = tmp_path / "deep-swe"
    _make_task(shard, "no-img", docker_image=None)
    with pytest.raises(SystemExit, match="no \\[environment\\] docker_image"):
        _task_image_ref(shard, "no-img")


def test_generate_plan_shape(tmp_path: Path) -> None:
    shard = tmp_path / "deep-swe"
    _make_task(shard, "t1", docker_image="public.ecr.aws/x/img:t1")
    _make_task(shard, "t2", docker_image="public.ecr.aws/x/img:t2")

    plan = generate_plan(["t1", "t2"], shard_dir=shard, probe_sizes=False)
    assert plan["version"] == 1
    assert plan["name"] == "deep-swe-2-task"
    assert len(plan["entries"]) == 2
    e = plan["entries"][0]
    assert e["image_ref"] == "public.ecr.aws/x/img:t1"
    assert e["context_source"] == {"type": "registry"}
    # probe off -> conservative heuristic hint
    assert e["placement"]["size_hint_bytes"] == DEFAULT_SIZE_HINT_BYTES
    assert e["placement"]["size_hint_source"] == "heuristic"
