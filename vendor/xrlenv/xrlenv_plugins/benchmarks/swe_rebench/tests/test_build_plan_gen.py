"""Unit tests for the SWE-rebench build-plan generator (pure logic; no network)."""
from __future__ import annotations

from pathlib import Path

import pytest
from xrlenv_plugins.benchmarks.swe_rebench.build_plan_gen import (
    DEFAULT_SIZE_HINT_BYTES,
    _discover_all_tasks,
    _split_repo_tag,
    _task_image_ref,
    generate_plan,
    known_sizes_from_plan,
)


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        # the SWE-rebench upstream shape
        (
            "swerebench/sweb.eval.x86_64.berriai_1776_litellm-14715:latest",
            ("swerebench/sweb.eval.x86_64.berriai_1776_litellm-14715", "latest"),
        ),
        # a registry host:PORT prefix must not be mistaken for the tag
        ("reg:5011/swe-rebench/t", ("reg:5011/swe-rebench/t", "latest")),
        ("reg:5011/swe-rebench/t:v2", ("reg:5011/swe-rebench/t", "v2")),
        # plain, no tag
        ("busybox", ("busybox", "latest")),
    ],
)
def test_split_repo_tag(ref: str, expected: tuple[str, str]) -> None:
    assert _split_repo_tag(ref) == expected


def _make_task(shard: Path, name: str, *, docker_image: str | None) -> None:
    d = shard / name
    (d / "environment").mkdir(parents=True)
    body = "[environment]\ncpus = 1\n"
    if docker_image is not None:
        body += f'docker_image = "{docker_image}"\n'
    (d / "task.toml").write_text(body)


def test_discover_and_task_image_ref(tmp_path: Path) -> None:
    shard = tmp_path / "swe-rebench"
    _make_task(shard, "b-task", docker_image="swerebench/img:b")
    _make_task(shard, "a-task", docker_image="swerebench/img:a")
    (shard / "stray").mkdir()  # no task.toml -> ignored
    (shard / ".dataset-version.json").write_text("{}")  # dotfile -> ignored

    assert _discover_all_tasks(shard) == ["a-task", "b-task"]  # sorted
    assert _task_image_ref(shard, "a-task") == "swerebench/img:a"


def test_task_image_ref_fails_loud_without_docker_image(tmp_path: Path) -> None:
    """The load-bearing invariant: never synthesize a ref. A task whose repin
    never ran must fail loud, pointing at the repin stage."""
    shard = tmp_path / "swe-rebench"
    _make_task(shard, "no-img", docker_image=None)
    with pytest.raises(SystemExit, match="--stage repin"):
        _task_image_ref(shard, "no-img")


def test_task_image_ref_fails_loud_missing_task(tmp_path: Path) -> None:
    shard = tmp_path / "swe-rebench"
    shard.mkdir(parents=True)
    with pytest.raises(SystemExit, match="is the shard populated"):
        _task_image_ref(shard, "nope")


def test_generate_plan_registry_entries(tmp_path: Path) -> None:
    shard = tmp_path / "swe-rebench"
    _make_task(shard, "a", docker_image="swerebench/img:a")
    _make_task(shard, "b", docker_image="swerebench/img:b")

    plan = generate_plan(["a", "b"], shard_dir=shard, probe_sizes=False)

    assert plan["version"] == 1
    assert plan["name"] == "swe-rebench-2-task"
    assert [e["image_ref"] for e in plan["entries"]] == [
        "swerebench/img:a", "swerebench/img:b",
    ]
    for entry in plan["entries"]:
        assert entry["context_source"] == {"type": "registry"}
        assert entry["placement"]["size_hint_bytes"] == DEFAULT_SIZE_HINT_BYTES
        assert entry["placement"]["size_hint_source"] == "heuristic"
        assert entry["labels"]["xrlenv.benchmark"] == "swe-rebench"
    assert plan["entries"][0]["labels"]["xrlenv.task_id"] == "a"


def test_generate_plan_uses_probe_sizes_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful probe becomes size_hint_source=registry-probe; a miss falls
    back to the heuristic. Probing is stubbed — the tests stay network-free."""
    import xrlenv_plugins.benchmarks._dockerhub_probe as probe_mod

    monkeypatch.setattr(probe_mod, "announce_auth_status", lambda: None)
    monkeypatch.setattr(probe_mod, "print_probe_summary", lambda _n: None)
    monkeypatch.setattr(
        probe_mod, "probe_image_size",
        lambda repo, tag: 1234 if repo.endswith("hit") else None,
    )

    shard = tmp_path / "swe-rebench"
    _make_task(shard, "a", docker_image="swerebench/hit:latest")
    _make_task(shard, "b", docker_image="swerebench/miss:latest")

    plan = generate_plan(["a", "b"], shard_dir=shard, probe_sizes=True)
    hit, miss = plan["entries"]
    assert hit["placement"] == {
        "preferred_home_count": 1,
        "size_hint_bytes": 1234,
        "size_hint_source": "registry-probe",
    }
    assert miss["placement"]["size_hint_bytes"] == DEFAULT_SIZE_HINT_BYTES
    assert miss["placement"]["size_hint_source"] == "heuristic"


def test_known_sizes_from_plan_keeps_only_measured_entries(tmp_path: Path) -> None:
    """Reusing a heuristic size would freeze the rate-limit fallback in place and
    make the resumable-probe loop never converge — so only registry-probe wins."""
    import yaml

    plan = tmp_path / "plan.yaml"
    plan.write_text(yaml.safe_dump({"entries": [
        {"image_ref": "probed:1",
         "placement": {"size_hint_bytes": 111, "size_hint_source": "registry-probe"}},
        {"image_ref": "heuristic:1",
         "placement": {"size_hint_bytes": 222, "size_hint_source": "heuristic"}},
        {"image_ref": "calibrated:1",
         "placement": {"size_hint_bytes": 333, "size_hint_source": "cluster-reported"}},
    ]}))
    assert known_sizes_from_plan(plan) == {"probed:1": 111}


@pytest.mark.parametrize("body", ["", "not-a-mapping", "entries: [1, 2]\n"])
def test_known_sizes_from_plan_tolerates_junk(tmp_path: Path, body: str) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text(body)
    assert known_sizes_from_plan(plan) == {}


def test_known_sizes_from_plan_missing_file(tmp_path: Path) -> None:
    assert known_sizes_from_plan(tmp_path / "nope.yaml") == {}


def test_generate_plan_reuses_known_sizes_and_skips_reprobing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resumable-probe contract: a ref whose size is already known must not
    be probed again (that is what lets successive runs beat the rate limit)."""
    import xrlenv_plugins.benchmarks._dockerhub_probe as probe_mod

    probed: list[str] = []

    def _fake_probe(repo: str, tag: str) -> int:
        probed.append(f"{repo}:{tag}")
        return 999

    monkeypatch.setattr(probe_mod, "announce_auth_status", lambda: None)
    monkeypatch.setattr(probe_mod, "print_probe_summary", lambda _n: None)
    monkeypatch.setattr(probe_mod, "probe_image_size", _fake_probe)

    shard = tmp_path / "swe-rebench"
    _make_task(shard, "known", docker_image="swerebench/known:latest")
    _make_task(shard, "unknown", docker_image="swerebench/unknown:latest")

    plan = generate_plan(
        ["known", "unknown"], shard_dir=shard, probe_sizes=True,
        known_sizes={"swerebench/known:latest": 4242},
    )

    assert probed == ["swerebench/unknown:latest"]
    by_ref = {e["image_ref"]: e["placement"] for e in plan["entries"]}
    assert by_ref["swerebench/known:latest"]["size_hint_bytes"] == 4242
    assert by_ref["swerebench/known:latest"]["size_hint_source"] == "registry-probe"
    assert by_ref["swerebench/unknown:latest"]["size_hint_bytes"] == 999


def test_generate_plan_output_loads_as_a_real_build_plan(tmp_path: Path) -> None:
    """The generated dict must satisfy xrlenv's own BuildPlan model — a shape
    drift here would only surface at `xrlenv build apply` time."""
    import yaml
    from xrlenv.control.build_plan import load_build_plan

    shard = tmp_path / "swe-rebench"
    _make_task(shard, "a", docker_image="swerebench/img:a")
    plan = generate_plan(["a"], shard_dir=shard, probe_sizes=False)

    out = tmp_path / "plan.yaml"
    out.write_text(yaml.safe_dump(plan, sort_keys=False))
    parsed = load_build_plan(out)

    assert len(parsed.entries) == 1
    assert parsed.entries[0].image_ref == "swerebench/img:a"
    assert parsed.entries[0].context_source.type == "registry"
