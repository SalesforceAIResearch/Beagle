"""Unit tests for the LHTB build-plan generator (pure logic; no network).

The generator emits ONE plan — the single source of truth for where each task's
image comes from. Each task is typed by its own ``task.toml`` ``docker_image``: a
REBUILD task repinned to a private registry becomes a ``type: local`` build entry
(+ compose sidecars); every prebuilt docker.io task becomes a ``type: registry``
entry. These tests pin that typing against synthetic shards.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from xrlenv_plugins.benchmarks.lhtb.build_plan_gen import (
    DEFAULT_LOCAL_SIZE_HINT_BYTES,
    DEFAULT_SIZE_HINT_BYTES,
    _discover_all_tasks,
    _split_repo_tag,
    _task_image_ref,
    generate_plan,
)

# A minimal chess-mate-shaped multi-service task: a main service with no build
# (harbor fills it) + a `game` sidecar that builds `.` with a CUSTOM dockerfile.
_COMPOSE_MULTISERVICE = """\
services:
  main:
    depends_on: {game: {condition: service_healthy}}
  game:
    build:
      context: .
      dockerfile: Dockerfile.game
"""

# A private-registry ref is what `build_cache --stage all --registry` repins a
# REBUILD task's docker_image to; a bare `zli12321/...` ref is a prebuilt docker.io
# image (no private-registry namespace → type: registry).
_PRIV = "node-host:5011/lhtb"


@pytest.mark.parametrize(("ref", "expected"), [
    ("zli12321/lhtb-2048:20260615", ("zli12321/lhtb-2048", "20260615")),
    ("zhongzhi660/lhtb-x:20260709", ("zhongzhi660/lhtb-x", "20260709")),
    ("busybox", ("busybox", "latest")),
])
def test_split_repo_tag(ref, expected) -> None:
    assert _split_repo_tag(ref) == expected


def _mk(shard: Path, name: str, *, docker_image: str | None) -> None:
    d = shard / name
    d.mkdir(parents=True)
    body = "[environment]\n"
    if docker_image is not None:
        body += f'docker_image = "{docker_image}"\n'
    (d / "task.toml").write_text(body)


def test_discover_and_task_image_ref(tmp_path: Path) -> None:
    shard = tmp_path / "lhtb"
    _mk(shard, "sokoban", docker_image="zli12321/lhtb-sokoban:x")
    _mk(shard, "2048", docker_image="zli12321/lhtb-2048:x")
    (shard / "stray").mkdir()
    assert _discover_all_tasks(shard) == ["2048", "sokoban"]
    assert _task_image_ref(shard, "2048") == "zli12321/lhtb-2048:x"


def test_task_image_ref_fails_loud_without_docker_image(tmp_path: Path) -> None:
    shard = tmp_path / "lhtb"
    _mk(shard, "no-img", docker_image=None)
    with pytest.raises(SystemExit, match="no \\[environment\\] docker_image"):
        _task_image_ref(shard, "no-img")


def test_task_image_ref_expands_registry_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repinned task's host-agnostic placeholder is expanded from .env at plan-gen time
    → a concrete ref for build/push/warm (GUIDELINE §5.3.1)."""
    monkeypatch.setenv("XRLENV_PRIVATE_REGISTRY_HOST", "node-host")
    monkeypatch.setenv("XRLENV_PRIVATE_REGISTRY_PORT", "5011")
    shard = tmp_path / "lhtb"
    _mk(shard, "chess-mate",
        docker_image="${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}"
        "/lhtb/chess-mate:main")
    assert _task_image_ref(shard, "chess-mate") == "node-host:5011/lhtb/chess-mate:main"


def test_task_image_ref_fails_loud_on_unresolved_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XRLENV_PRIVATE_REGISTRY_HOST", raising=False)
    shard = tmp_path / "lhtb"
    _mk(shard, "chess-mate", docker_image="${XRLENV_PRIVATE_REGISTRY_HOST}:5011/lhtb/x:main")
    with pytest.raises(SystemExit, match="unresolved registry placeholder"):
        _task_image_ref(shard, "chess-mate")


# ── type: registry entries (prebuilt docker.io images) ───────────────────────


def test_generate_plan_registry_entry_shape(tmp_path: Path) -> None:
    shard = tmp_path / "lhtb"
    _mk(shard, "2048", docker_image="zli12321/lhtb-2048:x")
    # not a rebuild task → type: registry, carrying the authoritative docker.io ref.
    plan = generate_plan(
        ["2048"], shard_dir=shard, probe_sizes=False, rebuild_tasks=frozenset(),
    )
    assert plan["name"] == "lhtb-1-task"
    e = plan["entries"][0]
    assert e["image_ref"] == "zli12321/lhtb-2048:x"
    assert e["context_source"] == {"type": "registry"}
    assert e["placement"]["size_hint_bytes"] == DEFAULT_SIZE_HINT_BYTES
    assert e["placement"]["size_hint_source"] == "heuristic"


def test_generate_plan_rebuild_task_still_on_docker_io_is_registry(tmp_path: Path) -> None:
    # A REBUILD task whose docker_image is NOT repinned (still docker.io, the
    # out-of-box path) has no private namespace → type: registry, not local.
    shard = tmp_path / "lhtb"
    _mk(shard, "chess-mate", docker_image="zli12321/lhtb-chess-mate:20260615")
    plan = generate_plan(
        ["chess-mate"], shard_dir=shard, probe_sizes=False,
        rebuild_tasks=frozenset({"chess-mate"}),
    )
    assert [e["context_source"]["type"] for e in plan["entries"]] == ["registry"]


# ── type: local entries (the images we build ourselves, post-repin) ──────────


def _mk_compose_task(
    shard: Path, name: str, compose: str, docker_image: str, *,
    dockerfiles: tuple[str, ...],
) -> Path:
    """A shard task with task.toml + an environment/ holding a compose + Dockerfiles."""
    _mk(shard, name, docker_image=docker_image)
    env = shard / name / "environment"
    env.mkdir(parents=True)
    (env / "docker-compose.yaml").write_text(compose)
    for df in dockerfiles:
        (env / df).write_text("FROM python:3.11-slim\n")
    return env


def _mk_buildable(shard: Path, name: str, docker_image: str) -> Path:
    """A single-service buildable task: task.toml + environment/Dockerfile, no compose."""
    _mk(shard, name, docker_image=docker_image)
    env = shard / name / "environment"
    env.mkdir(parents=True)
    (env / "Dockerfile").write_text("FROM python:3.11-slim\n")
    return env


def test_generate_plan_local_multi_service_main_and_sidecar(tmp_path: Path) -> None:
    shard = tmp_path / "lhtb"
    _mk_compose_task(shard, "chess-mate", _COMPOSE_MULTISERVICE,
                     f"{_PRIV}/chess-mate:main", dockerfiles=("Dockerfile", "Dockerfile.game"))
    plan = generate_plan(
        ["chess-mate"], shard_dir=shard, probe_sizes=False,
        rebuild_tasks=frozenset({"chess-mate"}),
    )
    entries = {e["image_ref"]: e for e in plan["entries"]}
    # main image = the repinned docker_image; game sidecar (custom dockerfile ->
    # distinct <task>-<service> ref) derived from that same repinned ref. Both local.
    assert set(entries) == {f"{_PRIV}/chess-mate:main", f"{_PRIV}/chess-mate-game:main"}
    main = entries[f"{_PRIV}/chess-mate:main"]
    game = entries[f"{_PRIV}/chess-mate-game:main"]
    assert main["context_source"]["type"] == "local"
    assert main["context_source"]["dockerfile"] == "Dockerfile"
    assert game["context_source"]["dockerfile"] == "Dockerfile.game"
    assert main["context_source"]["path"] == game["context_source"]["path"]  # same `.`
    assert game["labels"]["xrlenv.compose_service"] == "game"
    # label is the benchmark name, NOT the derived registry namespace.
    assert main["labels"]["xrlenv.benchmark"] == "lhtb"
    assert main["placement"]["size_hint_bytes"] == DEFAULT_LOCAL_SIZE_HINT_BYTES


def test_generate_plan_local_single_service_main_only(tmp_path: Path) -> None:
    # a baked-defect rebuild task (single-service) → just the main image, no sidecar.
    shard = tmp_path / "lhtb"
    _mk_buildable(shard, "duckdb-optimizer-closure", f"{_PRIV}/duckdb-optimizer-closure:main")
    plan = generate_plan(
        ["duckdb-optimizer-closure"], shard_dir=shard, probe_sizes=False,
        rebuild_tasks=frozenset({"duckdb-optimizer-closure"}),
    )
    assert [e["image_ref"] for e in plan["entries"]] == \
        [f"{_PRIV}/duckdb-optimizer-closure:main"]
    assert plan["entries"][0]["context_source"]["type"] == "local"


def test_generate_plan_mixed_local_and_registry(tmp_path: Path) -> None:
    # THE point: one plan, a genuine mixture — the single source of truth.
    shard = tmp_path / "lhtb"
    _mk(shard, "2048", docker_image="zli12321/lhtb-2048:x")  # docker.io -> registry
    _mk_buildable(shard, "duckdb-optimizer-closure", f"{_PRIV}/duckdb-optimizer-closure:main")
    _mk_compose_task(shard, "chess-mate", _COMPOSE_MULTISERVICE,
                     f"{_PRIV}/chess-mate:main", dockerfiles=("Dockerfile", "Dockerfile.game"))
    plan = generate_plan(
        ["2048", "duckdb-optimizer-closure", "chess-mate"], shard_dir=shard,
        probe_sizes=False,
        rebuild_tasks=frozenset({"duckdb-optimizer-closure", "chess-mate"}),
    )
    by_type: dict[str, set[str]] = {"local": set(), "registry": set()}
    for e in plan["entries"]:
        by_type[e["context_source"]["type"]].add(e["image_ref"])
    assert by_type["registry"] == {"zli12321/lhtb-2048:x"}
    assert by_type["local"] == {
        f"{_PRIV}/duckdb-optimizer-closure:main",
        f"{_PRIV}/chess-mate:main",
        f"{_PRIV}/chess-mate-game:main",
    }


def test_generate_plan_local_missing_main_dockerfile(tmp_path: Path) -> None:
    shard = tmp_path / "lhtb"
    # repinned + compose present, but no environment/Dockerfile for the main image.
    _mk_compose_task(shard, "chess-mate", _COMPOSE_MULTISERVICE,
                     f"{_PRIV}/chess-mate:main", dockerfiles=("Dockerfile.game",))
    with pytest.raises(SystemExit, match=r"no .*/Dockerfile"):
        generate_plan(["chess-mate"], shard_dir=shard, probe_sizes=False,
                      rebuild_tasks=frozenset({"chess-mate"}))


def test_generate_plan_local_missing_sidecar_dockerfile(tmp_path: Path) -> None:
    shard = tmp_path / "lhtb"
    # main Dockerfile present, but the game sidecar's Dockerfile.game is missing.
    _mk_compose_task(shard, "chess-mate", _COMPOSE_MULTISERVICE,
                     f"{_PRIV}/chess-mate:main", dockerfiles=("Dockerfile",))
    with pytest.raises(SystemExit, match=r"Dockerfile\.game.*missing"):
        generate_plan(["chess-mate"], shard_dir=shard, probe_sizes=False,
                      rebuild_tasks=frozenset({"chess-mate"}))
