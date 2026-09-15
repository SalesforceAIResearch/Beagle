"""Real Docker check: per-project cache isolation, with intra-project sharing.

Prerequisites: Linux Docker daemon, Compose v2, and ``docker pull alpine:3.22``.
From vendor/xrlenv: python -m pytest tests/smoke/test_compose_volume_isolation.py -q
Only randomly named test projects and their own volumes are removed.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
import yaml
from xrlenv.control.compose_policy import vet_compose_project
from xrlenv.control.compose_prepare import prepare_compose
from xrlenv.node.raw_compose import ComposeProjectRunner, ShellResult

pytestmark = pytest.mark.docker


@pytest.fixture
def docker_cli() -> str:
    binary = shutil.which("docker")
    if not binary:
        pytest.skip("Docker CLI unavailable")
    for args in (["info", "--format", "{{.OSType}}"], ["compose", "version"],
                 ["image", "inspect", "alpine:3.22"]):
        result = subprocess.run([binary, *args], capture_output=True, text=True, timeout=30)
        if result.returncode:
            pytest.skip(f"Docker prerequisite unavailable: {' '.join(args)}")
        if args[0] == "info" and result.stdout.strip() != "linux":
            pytest.skip("Requires Linux containers")
    return binary


async def test_compose_cache_cannot_carry_messages_between_projects(docker_cli: str, tmp_path: Path) -> None:
    async def run(argv, *, timeout_s=None):
        result = await asyncio.to_thread(
            subprocess.run, [docker_cli, *argv[1:]],
            capture_output=True, text=True, timeout=timeout_s,
        )
        return ShellResult(rc=result.returncode, stdout=result.stdout, stderr=result.stderr)

    async def command(container, *args):
        result = await run(["docker", "exec", container, *args], timeout_s=30)
        assert result.rc == 0, result.stderr or result.stdout
        return result.stdout.strip()

    compose = {
        "services": {
            "main": {"image": "alpine:3.22", "command": ["sleep", "300"],
                     "network_mode": "none", "volumes": ["cache:/cache"]},
            "sidecar": {"image": "alpine:3.22", "command": ["sleep", "300"],
                        "network_mode": "none", "volumes_from": ["main"]},
        },
        "volumes": {"cache": None},
    }
    vet_compose_project(compose)
    runner = ComposeProjectRunner(run=run, root_dir=tmp_path)
    records = []
    try:
        for _ in range(2):
            rollout_id = uuid.uuid4().hex
            project = f"xrlenv-storage-test-{rollout_id}"
            prepared = prepare_compose(compose, [], rollout_id=rollout_id, project_name=project)
            records.append(await runner.up(
                project_name=project, compose_yaml=yaml.safe_dump(prepared.compose),
                up_timeout_s=90,
            ))
        first, second = records
        await command(first.main_container_id, "sh", "-c", "printf first > /cache/message")
        assert await command(first.service_container_ids["sidecar"], "cat", "/cache/message") == "first"
        await command(second.main_container_id, "test", "!", "-e", "/cache/message")
        await command(second.main_container_id, "sh", "-c", "printf second > /cache/message")
        assert await command(first.main_container_id, "cat", "/cache/message") == "first"
        # Teardown of one project must neither remove nor change the other's cache.
        await runner.down(project_name=first.project_name, project_dir=first.project_dir)
        records.remove(first)
        assert await command(second.service_container_ids["sidecar"], "cat", "/cache/message") == "second"
    finally:
        # Try each cleanup even if an earlier project fails to come down.
        results = await asyncio.gather(*(
            runner.down(project_name=rec.project_name, project_dir=rec.project_dir)
            for rec in records
        ), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
