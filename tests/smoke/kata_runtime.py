"""Opt-in, offline Beagle lifecycle smoke on an operator-configured Kata/QEMU host.

Run on that Linux host with permission to use its Docker socket and read QEMU's
/proc entries. Requires a pre-pulled Alpine image; makes no model/API calls.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import beagle as bgl
from beagle.agents.core.base import Runnable
from beagle.benchmarks import Benchmark, InBandGrader, register
from beagle.benchmarks.harness.drivers import DockerHarness
from beagle.config import RunConfig
from beagle.rollout.runtime import KataDockerRuntime
from beagle.types import RolloutStatus, Task, TaskContext, TaskResult


def vm_processes(container_id: str) -> list[int]:
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if argv and b"qemu-system" in argv[0] and any(
            container_id.encode() in arg for arg in argv
        ):
            found.append(int(proc.name))
    return found


class ProbeFailure(RuntimeError):
    """Intentional agent exception to exercise Runnable's finally block."""


class Probe(Runnable):
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.container_id = ""

    def install(self, handle, task_ctx, *, runtime):
        self.container_id = handle.container_id
        result = runtime.exec(handle, ["sh", "-c", "printf beagle-kata > /tmp/probe"])
        assert result.ok, result.stderr

    def run_in(self, handle, task, task_ctx, *, runtime):
        marker = runtime.exec(handle, ["cat", "/tmp/probe"])
        assert marker.ok and marker.stdout.strip() == "beagle-kata"
        kernel = runtime.exec(handle, ["uname", "-r"])
        assert kernel.ok and kernel.stdout.strip() != os.uname().release
        interfaces = runtime.exec(handle, ["ls", "/sys/class/net"])
        assert interfaces.ok and interfaces.stdout.split() == ["lo"]
        vms = vm_processes(handle.container_id)
        assert vms, "No associated QEMU process visible on this host"
        print(json.dumps({"container": handle.container_id, "qemu_pids": vms,
                          "guest_kernel": kernel.stdout.strip(), "network": "none",
                          "intentional_failure": self.fail}), flush=True)
        if self.fail:
            raise ProbeFailure("exercise cleanup after agent failure")
        # Timeout must be reported as 124; container teardown ends the guest process.
        assert runtime.exec(handle, ["sleep", "10"], timeout=1).returncode == 124
        return TaskResult(task_id=task.task_id, status=RolloutStatus.COMPLETED,
                          reward=1.0, resolved=True)


def check_removed(runtime: KataDockerRuntime, container_id: str) -> None:
    assert container_id, "Probe did not reach install"
    result = subprocess.run(
        ["docker", "--host", runtime.docker_host, "ps", "-aq", "--filter", f"id={container_id}"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    assert not result.stdout.strip(), "Container still present"
    deadline = time.monotonic() + 10
    while vm_processes(container_id) and time.monotonic() < deadline:
        time.sleep(0.25)
    assert not vm_processes(container_id), "QEMU still running after removal"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-host", default="unix:///var/run/docker.sock")
    parser.add_argument("--runtime-name", default="kata")
    parser.add_argument("--image", default="alpine:3.22")
    args = parser.parse_args()
    runtime = KataDockerRuntime(docker_host=args.docker_host, runtime_name=args.runtime_name)
    ctx = TaskContext(image=args.image, agent_timeout_s=120)
    for fail in (False, True):
        probe = Probe(fail=fail)
        task = Task(task_id=f"kata-probe-{fail}", benchmark="kata-smoke")
        try:
            result = DockerHarness().run(probe.rollout_binding(ctx), task, ctx, runtime=runtime)
        except ProbeFailure:
            assert fail
        else:
            assert not fail
            assert result.status == RolloutStatus.COMPLETED
            assert set(result.timing) == {"environment_setup", "agent_setup", "agent_execution"}
        check_removed(runtime, probe.container_id)
    print("PASS: Beagle acquire/install/exec/timeout/destroy; failure cleanup; QEMU termination")

    @register("kata-lifecycle-smoke")
    class ProbeBenchmark(Benchmark):
        def source(self):
            raise NotImplementedError("The smoke supplies its dataset directly")

        def harness(self, env_import_path=None):
            return DockerHarness()

        def grader(self):
            return InBandGrader()

    config = RunConfig.from_dict({
        "model": {"name": "unused"}, "agent": {"name": "monet"},
        "benchmark": {"name": "kata-lifecycle-smoke"}, "parallelism": 1,
        "runtime": {"kind": "kata", "options": {
            "docker_host": args.docker_host, "runtime_name": args.runtime_name}},
    })
    probe = Probe()
    with tempfile.TemporaryDirectory(prefix="beagle-kata-smoke-") as directory:
        result = bgl.evaluate(config, agent=probe, run_dir=directory, dataset=[(
            Task(task_id="evaluate", benchmark="kata-lifecycle-smoke"), ctx)])
        assert len(result.results) == 1 and result.results[0].resolved
        assert (Path(directory) / "run.json").is_file()
    check_removed(runtime, probe.container_id)
    print("PASS: evaluate config -> Runner -> DockerHarness -> Kata -> InBandGrader -> run.json")


if __name__ == "__main__":
    main()
