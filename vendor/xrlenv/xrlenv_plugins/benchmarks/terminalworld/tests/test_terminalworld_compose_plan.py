"""Multi-service compose awareness in the terminalworld build-plan generator.

A multi-service task with a **sub-directory** build context (like tw_188260's
``solr-node`` / ``ambari-server``) must yield an extra ``type: local`` build
entry per sub-context, named ``<id>-<service>:tag``, in addition to the task's
canonical ``<id>:tag`` entry. Single-service tasks and multi-service tasks whose
services all build from ``.`` (or use public images) yield exactly the one entry
they always have — no regression. Offline: a synthetic cache shard in tmp, no
FSx, no cluster.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from xrlenv_plugins.benchmarks.terminalworld import build_plan_gen as gen


def _write_task(
    shard: Path, task_id: str, *, compose: str | None = None,
    services: dict[str, str] | None = None,
) -> None:
    """Materialize a minimal task dir: ``environment/Dockerfile`` (+ optional
    ``docker-compose.yaml`` and per-service ``<svc>/Dockerfile``)."""
    env = shard / task_id / "environment"
    env.mkdir(parents=True)
    (env / "Dockerfile").write_text("FROM busybox\n")
    if compose is not None:
        (env / "docker-compose.yaml").write_text(compose)
    for svc_dir, dockerfile in (services or {}).items():
        d = env / svc_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / dockerfile).write_text("FROM busybox\n")


@pytest.fixture()
def shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cache"
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(root))
    return root / gen.SHARD


def _refs(plan: dict) -> list[str]:
    return [e["image_ref"] for e in plan["entries"]]


def test_single_service_task_one_entry(shard: Path) -> None:
    _write_task(shard, "tw_1")  # no compose
    plan = gen.generate_plan(["tw_1"])
    assert _refs(plan) == ["terminalworld-verified/tw_1:main"]


def test_multi_service_all_dot_context_no_extra_entries(shard: Path) -> None:
    # Peers building from ``.`` reuse the canonical image; a public sidecar is
    # pulled, not built → still exactly one build entry.
    compose = (
        "services:\n"
        "  main: {}\n"
        "  peer: {build: '.'}\n"
        "  db: {image: postgres:14}\n"
    )
    _write_task(shard, "tw_2", compose=compose)
    plan = gen.generate_plan(["tw_2"])
    assert _refs(plan) == ["terminalworld-verified/tw_2:main"]


def test_multi_service_subdir_contexts_add_entries(shard: Path) -> None:
    compose = (
        "services:\n"
        "  main: {}\n"
        "  solr-node: {build: {context: ./solr-node, dockerfile: Dockerfile}}\n"
        "  ambari-server: {build: ./ambari-server}\n"
    )
    _write_task(
        shard, "tw_3", compose=compose,
        services={"solr-node": "Dockerfile", "ambari-server": "Dockerfile"},
    )
    plan = gen.generate_plan(["tw_3"])
    assert _refs(plan) == [
        "terminalworld-verified/tw_3:main",
        "terminalworld-verified/tw_3-ambari-server:main",
        "terminalworld-verified/tw_3-solr-node:main",
    ]
    # the sub-context entries point at the sub-dir + carry a service label.
    by_ref = {e["image_ref"]: e for e in plan["entries"]}
    solr = by_ref["terminalworld-verified/tw_3-solr-node:main"]
    assert solr["context_source"]["path"].endswith("/tw_3/environment/solr-node")
    assert solr["labels"]["xrlenv.compose_service"] == "solr-node"


def test_subdir_service_custom_dockerfile_name(shard: Path) -> None:
    compose = (
        "services:\n"
        "  main: {}\n"
        "  svc: {build: {context: ./svc, dockerfile: Dockerfile.svc}}\n"
    )
    _write_task(shard, "tw_4", compose=compose, services={"svc": "Dockerfile.svc"})
    plan = gen.generate_plan(["tw_4"])
    svc = next(e for e in plan["entries"] if "svc" in e["image_ref"])
    assert svc["context_source"]["dockerfile"] == "Dockerfile.svc"


def test_missing_service_build_context_errors(shard: Path) -> None:
    compose = (
        "services:\n"
        "  main: {}\n"
        "  svc: {build: ./svc}\n"  # declared but no ./svc/Dockerfile written
    )
    _write_task(shard, "tw_5", compose=compose)
    with pytest.raises(SystemExit, match=r"svc.*missing"):
        gen.generate_plan(["tw_5"])


@pytest.mark.parametrize("context", ["../escape", "/etc", "../../host"])
def test_build_context_escape_rejected(shard: Path, context: str) -> None:
    # A compose build context that escapes the task environment dir must fail
    # loud before a type: local entry pointing outside it is emitted.
    compose = (
        "services:\n"
        "  main: {}\n"
        f"  svc: {{build: '{context}'}}\n"
    )
    _write_task(shard, "tw_6", compose=compose)
    with pytest.raises(SystemExit, match=r"escapes the task environment"):
        gen.generate_plan(["tw_6"])
