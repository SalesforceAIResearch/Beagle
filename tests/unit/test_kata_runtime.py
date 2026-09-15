from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from beagle.benchmarks.harness.drivers import HarborHarness, PierHarness
from beagle.rollout.runtime import (
    BindMount,
    ContainerHandle,
    ContainerResources,
    KataDockerRuntime,
    KataRuntimeError,
    RuntimeConfig,
    build_runtime,
)


def response(stdout="", rc=0):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr="daemon error" if rc else "")


INFO = response(json.dumps({"OSType": "linux", "Runtimes": {"kata": {}},
                            "MemoryLimit": True, "CPUCfsQuota": True}))
INSPECT = response(json.dumps([{"HostConfig": {"Runtime": "kata", "NetworkMode": "none"}}]))


def replies(monkeypatch, *results):
    pending = iter(results)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        result = next(pending)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr("beagle.rollout.runtime.kata_runtime.subprocess.run", run)
    return calls


def test_factory_preserves_explicit_socket_and_rejects_unknown_options():
    runtime = build_runtime(RuntimeConfig(kind="kata", options={"docker_host": "unix:///run/lab.sock"}))
    assert isinstance(runtime, KataDockerRuntime)
    assert runtime.docker_host == "unix:///run/lab.sock"
    with pytest.raises(TypeError):
        build_runtime(RuntimeConfig(kind="kata", options={"network": "host"}))


@pytest.mark.parametrize("info", [
    response(rc=1), response("not json"), response("null"),
    response(json.dumps({"OSType": "windows", "Runtimes": {"kata": {}}})),
    response(json.dumps({"OSType": "linux", "Runtimes": {"runc": {}}})),
    response(json.dumps({"OSType": "linux", "Runtimes": {"kata": {"path": "runc"}}})),
])
def test_unavailable_runtime_never_launches(monkeypatch, info):
    calls = replies(monkeypatch, info)
    with pytest.raises(KataRuntimeError):
        KataDockerRuntime().acquire(image="alpine:3.22")
    assert len(calls) == 1
    assert "run" not in calls[0][0]


@pytest.mark.parametrize("args", [
    ["--runtime", "runc"], ["--runtime=runc"], ["--network=host"],
    ["--privileged"], ["--volumes-from", "other"], ["--name", "shared"],
    ["--entrypoint", "sh", "--network", "bridge"], ["--entrypoint"],
])
def test_profile_cannot_be_overridden(monkeypatch, args):
    calls = replies(monkeypatch)
    with pytest.raises(ValueError):
        KataDockerRuntime().acquire(image="alpine", run_args=args)
    assert calls == []


@pytest.mark.parametrize("image", ["", "--runtime=runc"])
def test_image_cannot_inject_a_docker_option(monkeypatch, image):
    calls = replies(monkeypatch)
    with pytest.raises(ValueError):
        KataDockerRuntime().acquire(image=image)
    assert calls == []


def test_acquire_and_exec_use_same_daemon_without_secrets_in_argv(monkeypatch):
    calls = replies(monkeypatch, INFO, response("cid"), INSPECT, response("ok"))
    runtime = KataDockerRuntime(docker_host="unix:///run/lab.sock")
    handle = runtime.acquire(image="alpine", env={"TOKEN": "secret-value"},
                             run_args=["--entrypoint", ""], command=["sleep", "infinity"])
    assert runtime.exec(handle, ["true"], env={"TOKEN": "secret-value"}).ok
    for argv, _ in calls:
        assert argv[:3] == ["docker", "--host", "unix:///run/lab.sock"]
        assert "secret-value" not in " ".join(argv)
    launch, kwargs = calls[1]
    assert launch[launch.index("--runtime") + 1] == "kata"
    assert launch[launch.index("--network") + 1] == "none"
    assert kwargs["env"]["TOKEN"] == "secret-value"
    assert handle.container_id == "cid"


@pytest.mark.parametrize("failure", [response(rc=1), subprocess.TimeoutExpired("docker", 1)])
def test_failed_launch_is_cleaned_up_by_generated_name(monkeypatch, failure):
    calls = replies(monkeypatch, INFO, failure, response(), response())
    with pytest.raises((KataRuntimeError, subprocess.TimeoutExpired)):
        KataDockerRuntime().acquire(image="alpine")
    launch = calls[1][0]
    name = launch[launch.index("--name") + 1]
    assert calls[2][0][-3:] == ["rm", "-f", name]
    assert calls[3][0][-1] == f"name=^/{name}$"


def test_unverified_launch_is_destroyed(monkeypatch):
    bad = response(json.dumps([{"HostConfig": {"Runtime": "runc", "NetworkMode": "none"}}]))
    calls = replies(monkeypatch, INFO, response("cid"), bad, response(), response())
    with pytest.raises(KataRuntimeError, match="does not match"):
        KataDockerRuntime().acquire(image="alpine")
    assert calls[-2][0][-3:] == ["rm", "-f", "cid"]


def test_cleanup_failure_retains_handle_and_can_be_retried(monkeypatch):
    replies(monkeypatch, INFO, response(rc=1), response(rc=1), response("still-present"))
    runtime = KataDockerRuntime()
    with pytest.raises(KataRuntimeError, match="cleanup") as exc:
        runtime.acquire(image="alpine")
    handle = exc.value.handle
    assert handle and handle.container_id
    calls = replies(monkeypatch, response(), response())
    runtime.destroy(handle)
    assert handle.container_id == ""
    runtime.destroy(handle)
    assert len(calls) == 2


def test_known_id_cleanup_does_not_assume_original_name(monkeypatch):
    calls = replies(monkeypatch, response(), response())
    KataDockerRuntime().destroy(ContainerHandle(container_id="cid", name="old-name"))
    assert calls[-1][0][-1] == "id=cid"


@pytest.mark.parametrize("verification", [response(rc=1), subprocess.TimeoutExpired("docker", 30)])
def test_cleanup_never_treats_daemon_failure_as_absence(monkeypatch, verification):
    replies(monkeypatch, response(), verification)
    handle = ContainerHandle(container_id="cid", name="n")
    with pytest.raises(KataRuntimeError):
        KataDockerRuntime().destroy(handle)
    assert handle.container_id == "cid"


def test_writable_host_mount_rejected(monkeypatch):
    calls = replies(monkeypatch)
    with pytest.raises(ValueError, match="read-only"):
        KataDockerRuntime().acquire(image="alpine", mounts=[BindMount(
            host_path=Path("/shared"), container_path="/data", read_only=False)])
    assert calls == []


@pytest.mark.parametrize("resources", [ContainerResources(cpu_limit=1), ContainerResources(mem_limit_bytes=1024)])
def test_unsupported_resource_limits_fail_before_launch(monkeypatch, resources):
    calls = replies(monkeypatch, response(json.dumps({"OSType": "linux", "Runtimes": {"kata": {}}})))
    with pytest.raises(KataRuntimeError, match="cannot enforce"):
        KataDockerRuntime().acquire(image="alpine", resources=resources)
    assert len(calls) == 1


@pytest.mark.parametrize("harness", [HarborHarness, PierHarness])
@pytest.mark.parametrize("override", [None, "custom:Environment"])
def test_native_harness_cannot_silently_bypass_kata(harness, override):
    with pytest.raises(ValueError, match="does not support"):
        harness(runtime_kind="kata", env_import_path=override)


def test_runner_rejects_runtime_override_before_starting_work():
    from beagle.rollout.runner import Runner
    from beagle.rollout.runtime import LocalDockerRuntime

    with pytest.raises(ValueError, match="refusing a runtime override"):
        Runner(LocalDockerRuntime()).run(
            object(), [], config=SimpleNamespace(runtime=SimpleNamespace(kind="kata")))


@pytest.mark.parametrize("docker_harness", [False, True])
def test_runner_rejects_harness_or_external_grader_before_rollout(monkeypatch, tmp_path, docker_harness):
    from beagle.benchmarks.harness.drivers import DockerHarness
    from beagle.config import RunConfig
    from beagle.rollout.runner import Runner
    from beagle.types import Task, TaskContext

    harness = DockerHarness() if docker_harness else object()
    bench = SimpleNamespace(harness=lambda: harness, grader=lambda: object())
    monkeypatch.setattr("beagle.benchmarks.get", lambda name: bench)
    config = RunConfig.from_dict({
        "model": {"name": "unused"}, "agent": {"name": "monet"},
        "benchmark": {"name": "probe"}, "runtime": {"kind": "kata"},
    })
    expected = "InBandGrader" if docker_harness else "DockerHarness"
    with pytest.raises(TypeError, match=expected):
        Runner(KataDockerRuntime(), results_root=tmp_path).run(
            object(), [(Task(task_id="t", benchmark="probe"), TaskContext(image="alpine"))],
            config=config)
