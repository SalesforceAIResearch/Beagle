"""Local Docker cleanup regression; opt in with pytest on this file.

Requires a running Linux Docker daemon and pulls alpine:3.22 if missing.
No model calls or XRLEnv cluster are needed.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from beagle.rollout.runtime.runtime import ContainerCleanupError, LocalDockerRuntime


@pytest.mark.docker
def test_failed_cleanup_retains_live_container_then_retry_removes_it(monkeypatch) -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is unavailable")
    try:
        info = subprocess.run(
            ["docker", "info", "--format", "{{.OSType}}"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("Docker daemon is unavailable")
    if info.returncode != 0 or info.stdout.strip() != "linux":
        pytest.skip("A running Linux Docker daemon is required")

    runtime = LocalDockerRuntime()
    handle = runtime.acquire(image="alpine:3.22", command=["sleep", "infinity"])
    cid = handle.container_id
    real_run = subprocess.run

    def fail_teardown(argv, **kwargs):
        if argv in (["docker", "stop", cid], ["docker", "rm", "-f", cid]):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="injected failure")
        return real_run(argv, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr("beagle.rollout.runtime.runtime.subprocess.run", fail_teardown)
            with pytest.raises(ContainerCleanupError):
                runtime.destroy(handle)
        assert handle.container_id == cid
        assert runtime.exec(handle, ["true"]).ok

        runtime.destroy(handle)
        assert handle.container_id == ""
        # Verify independently from the runtime, including stopped containers.
        remaining = real_run(
            ["docker", "ps", "--all", "--quiet", "--no-trunc", "--filter", f"id={cid}"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        assert not remaining.stdout.strip()
        runtime.destroy(handle)
    finally:
        # Keep the original ID even when exercising the old, prematurely-cleared handle.
        real_run(
            ["docker", "rm", "-f", cid],
            capture_output=True, text=True, timeout=30, check=False,
        )
