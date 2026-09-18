from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from beagle.agents.core.base import Runnable
from beagle.rollout.runtime.runtime import (
    ContainerCleanupError,
    ContainerHandle,
    LocalDockerRuntime,
)
from beagle.types import RolloutStatus, Task, TaskContext, TaskResult


def test_exec_passes_environment_values_out_of_argv(monkeypatch) -> None:
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["env"] = kwargs.get("env")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("beagle.rollout.runtime.runtime.subprocess.run", fake_run)
    secret = "sk-secret-that-must-not-reach-ps"
    result = LocalDockerRuntime().exec(
        ContainerHandle(container_id="cid", name="n"),
        ["sh", "-lc", "true"],
        env={"BEAGLE_GATEWAY_CONFIG": secret},
    )

    assert result.ok
    assert secret not in " ".join(seen["argv"])
    assert seen["argv"][:4] == ["docker", "exec", "-e", "BEAGLE_GATEWAY_CONFIG"]
    assert seen["env"]["BEAGLE_GATEWAY_CONFIG"] == secret


def _result(returncode=0, *, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _docker_responses(monkeypatch, *responses):
    """Inject Docker failures without requiring a daemon or touching containers."""
    pending = iter(responses)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        response = next(pending)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("beagle.rollout.runtime.runtime.subprocess.run", fake_run)
    return calls


def test_destroy_confirms_removal_before_clearing_handle(monkeypatch) -> None:
    handle = ContainerHandle(container_id="cid", name="n")
    calls = []

    def fake_run(argv, **kwargs):
        # Even a successful rm does not invalidate the handle until verification completes.
        assert handle.container_id == "cid"
        calls.append(argv)
        return _result()

    monkeypatch.setattr("beagle.rollout.runtime.runtime.subprocess.run", fake_run)
    runtime = LocalDockerRuntime()
    runtime.destroy(handle)

    assert handle.container_id == ""
    assert calls[-1] == ["docker", "ps", "--all", "--quiet", "--no-trunc", "--filter", "id=cid"]
    count = len(calls)
    runtime.destroy(handle)
    assert len(calls) == count


def test_destroy_keeps_failed_handle_for_retry(monkeypatch) -> None:
    handle = ContainerHandle(container_id="cid", name="n")
    runtime = LocalDockerRuntime()
    _docker_responses(
        monkeypatch, _result(1, stderr="stop failed"),
        _result(1, stderr="remove failed"), _result(stdout="cid\n"),
    )

    with pytest.raises(ContainerCleanupError, match="cid") as failure:
        runtime.destroy(handle)
    assert failure.value.container_id == "cid"
    assert handle.container_id == "cid"

    _docker_responses(monkeypatch, _result(), _result(), _result())
    runtime.destroy(handle)
    assert handle.container_id == ""


@pytest.mark.parametrize("stop_failure", [
    _result(1, stderr="stop failed"),
    subprocess.TimeoutExpired(["docker", "stop", "cid"], 30),
    OSError("stop could not start"),
])
def test_destroy_attempts_force_removal_after_stop_failure(monkeypatch, stop_failure) -> None:
    handle = ContainerHandle(container_id="cid", name="n")
    calls = _docker_responses(monkeypatch, stop_failure, _result(), _result())

    LocalDockerRuntime().destroy(handle)

    assert ["docker", "rm", "-f", "cid"] in calls
    assert handle.container_id == ""


@pytest.mark.parametrize("remove_failure", [
    _result(1, stderr="No such container: cid"),
    subprocess.TimeoutExpired(["docker", "rm", "-f", "cid"], 30),
    OSError("lost connection"),
])
def test_destroy_accepts_absence_after_ambiguous_removal(monkeypatch, remove_failure) -> None:
    # A timeout or another actor's cleanup can leave the container already removed.
    handle = ContainerHandle(container_id="cid", name="n")
    _docker_responses(monkeypatch, _result(1), remove_failure, _result())

    LocalDockerRuntime().destroy(handle)

    assert handle.container_id == ""


@pytest.mark.parametrize("verification", [
    _result(stdout="cid\n"),  # A stopped container also counts as not removed (--all).
    _result(1, stderr="daemon unavailable"),
    subprocess.TimeoutExpired(["docker", "ps"], 30),
    OSError("docker not found"),
])
def test_destroy_does_not_confirm_unverified_removal(monkeypatch, verification) -> None:
    handle = ContainerHandle(container_id="cid", name="n")
    _docker_responses(monkeypatch, _result(), _result(), verification)

    with pytest.raises(ContainerCleanupError, match="cid"):
        LocalDockerRuntime().destroy(handle)

    assert handle.container_id == "cid"


def test_destroy_already_cleared_handle_does_not_call_docker(monkeypatch) -> None:
    calls = _docker_responses(monkeypatch)

    LocalDockerRuntime().destroy(ContainerHandle(container_id="", name="n"))

    assert calls == []


@pytest.mark.parametrize("execution_fails", [False, True])
def test_agent_lifecycle_propagates_cleanup_failure(monkeypatch, execution_fails) -> None:
    class TestAgent(Runnable):
        def install(self, handle, task_ctx, *, runtime):
            pass

        def run_in(self, handle, task, task_ctx, *, runtime):
            if execution_fails:
                raise ValueError("agent execution failed")
            return TaskResult(task_id=task.task_id, status=RolloutStatus.COMPLETED)

    handle = ContainerHandle(container_id="cid", name="n")
    runtime = LocalDockerRuntime()
    monkeypatch.setattr(runtime, "acquire", lambda **kwargs: handle)
    _docker_responses(monkeypatch, _result(1), _result(1), _result(stdout="cid\n"))

    # An otherwise completed task must not hide an unconfirmed cleanup. If the
    # task itself failed, Python's exception chain must retain that original error.
    with pytest.raises(ContainerCleanupError) as failure:
        TestAgent().run(Task(task_id="task"), TaskContext(image="test"), runtime=runtime)

    assert handle.container_id == "cid"
    if execution_fails:
        assert isinstance(failure.value.__context__, ValueError)
        assert str(failure.value.__context__) == "agent execution failed"


def test_cleanup_error_is_importable_from_the_package() -> None:
    """The contract tells callers not to suppress this error, so they must be able to NAME it.

    It was exported from the implementation module but not re-exported from the package, unlike
    every one of its siblings -- so the documented public import raised ImportError and a caller
    wanting to catch it had to reach past the package into `beagle.rollout.runtime.runtime`.
    """
    import beagle.rollout.runtime as pkg
    from beagle.rollout.runtime import ContainerCleanupError as FromPackage
    from beagle.rollout.runtime.runtime import ContainerCleanupError as FromModule

    assert FromPackage is FromModule
    assert "ContainerCleanupError" in pkg.__all__
