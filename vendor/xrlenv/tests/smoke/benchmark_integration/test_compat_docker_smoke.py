"""Real-Docker smoke for the docker-py drop-in.

Confirms ``xrlenv.from_env() → real container → exec_run → put_archive
→ remove`` works end-to-end on a real local Docker daemon. Skipped
automatically when no daemon is reachable.

This sits one tier above ``tests/unit/compat/test_compat_docker_shape.py``:
the unit tests verify the daemon-free shape (issubclass, signature,
lazy-import); this validates the actual round-trip via a live engine.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import tarfile

import docker
import pytest
from xrlenv.compat.docker_client import from_env

_IMAGE = "alpine:latest"


def _docker_reachable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


@pytest.fixture(scope="module")
def _ensure_image() -> None:
    if not _docker_reachable():
        pytest.skip("docker daemon not reachable")
    proc = subprocess.run(
        ["docker", "image", "inspect", _IMAGE],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pull = subprocess.run(
            ["docker", "pull", _IMAGE], capture_output=True, text=True,
        )
        if pull.returncode != 0:
            pytest.skip(f"could not pull {_IMAGE}: {pull.stderr}")


pytestmark = pytest.mark.skipif(
    not _docker_reachable(), reason="docker daemon not reachable",
)


def test_drop_in_runs_real_container_end_to_end(_ensure_image) -> None:
    """The architectural validation in code form: same diff a
    consumer would make — swap ``docker.from_env()`` for
    ``xrlenv.from_env()`` — and end up with a real container on
    the local engine, drivable through the standard docker-py
    surface (run / exec_run / put_archive / remove).
    """
    client = from_env()
    assert isinstance(client, docker.DockerClient)

    container = None
    try:
        container = client.containers.run(
            _IMAGE, command=["sleep", "60"], detach=True,
        )
        assert container.id

        result = container.exec_run(["sh", "-c", "echo hello-from-xrlenv"])
        assert result.exit_code == 0
        assert b"hello-from-xrlenv" in result.output

        # put_archive: write a tar in, read it back via exec.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            payload = b"contents-of-marker-file\n"
            info = tarfile.TarInfo(name="marker.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        container.put_archive("/tmp", buf.getvalue())

        readback = container.exec_run(["cat", "/tmp/marker.txt"])
        assert readback.exit_code == 0
        assert b"contents-of-marker-file" in readback.output

    finally:
        if container is not None:
            with contextlib.suppress(Exception):
                container.remove(force=True)


def test_streaming_exec_pattern_works(_ensure_image) -> None:
    """SWE-bench's harness uses the low-level
    ``container.client.api.exec_create + exec_start(stream=True)``
    pattern instead of ``container.exec_run``. Pin that path works
    against our drop-in because XrlenvAPIClient inherits from
    docker.APIClient with a real super().__init__() in LocalDocker
    mode."""
    client = from_env()
    container = None
    try:
        container = client.containers.run(
            _IMAGE, command=["sleep", "30"], detach=True,
        )
        # Same call shape SWE-bench's docker_utils.exec_run_with_timeout uses.
        exec_id = client.api.exec_create(
            container.id, ["sh", "-c", "for i in 1 2 3; do echo line-$i; done"],
        )["Id"]
        chunks: list[bytes] = []
        for chunk in client.api.exec_start(exec_id, stream=True):
            chunks.append(chunk)
        output = b"".join(chunks)
        for expected in (b"line-1", b"line-2", b"line-3"):
            assert expected in output
        # exit code via inspect — also the SWE-bench pattern.
        info = client.api.exec_inspect(exec_id)
        assert info["ExitCode"] == 0
    finally:
        if container is not None:
            with contextlib.suppress(Exception):
                container.remove(force=True)


def test_images_list_get_pull_round_trip(_ensure_image) -> None:
    """SWE-bench enumerates images via ``client.images.list(all=True)``
    and looks them up via ``client.images.get(name)``. Pin both work
    through the drop-in (inherited from docker-py against the local
    daemon)."""
    client = from_env()
    img = client.images.get(_IMAGE)
    assert _IMAGE in img.tags

    listed = client.images.list(all=True)
    listed_tags = {tag for i in listed for tag in i.tags}
    assert _IMAGE in listed_tags
