"""Experimental local Kata/QEMU runtime for prebuilt, offline workloads.

The operator installs and configures Kata on the Docker host. This adapter
requires an explicit Kata runtime registration and never falls back to runc.
No host-side nsenter/iptables egress mechanism is assumed to work for a VM.
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

from beagle.rollout.runtime.runtime import (
    ContainerHandle,
    ContainerResources,
    LocalDockerRuntime,
    _resource_run_flags,
)
from beagle.rollout.runtime.transport import BindMount


class KataRuntimeError(RuntimeError):
    """A Kata launch or confirmed cleanup failed; ``handle`` permits cleanup retry."""

    def __init__(self, message: str, *, handle: ContainerHandle | None = None) -> None:
        super().__init__(message)
        self.handle = handle


class KataDockerRuntime(LocalDockerRuntime):
    """One Docker-created Kata sandbox per acquire, with networking disabled.

    Supports local Unix Docker sockets, read-only input mounts and an entrypoint
    override. Dependencies must already be in the image. Docker registration is
    an operator trust boundary, not cryptographic attestation of the hypervisor.
    """

    def __init__(self, *, docker_host: str = "unix:///var/run/docker.sock",
                 runtime_name: str = "kata") -> None:
        if not docker_host.startswith("unix:///"):
            raise ValueError("Kata's experimental local runtime requires an absolute unix:/// Docker socket")
        if not runtime_name or runtime_name in ("runc", "io.containerd.runc.v2"):
            raise ValueError("runtime_name must name an operator-configured Kata runtime")
        super().__init__(docker_host=docker_host)
        self.runtime_name = runtime_name

    def _check_runtime(self, resources: ContainerResources | None) -> None:
        result = subprocess.run(
            self._docker_argv("info", "--format", "{{json .}}"),
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode:
            raise KataRuntimeError("Cannot inspect the requested Docker daemon")
        try:
            info = json.loads(result.stdout)
            registration = info["Runtimes"][self.runtime_name]
            # Docker may expose only {} for shim-v2 registrations. The alias
            # remains operator-configured; do not call this VM attestation.
            path = registration.get("path", "")
            if info["OSType"] != "linux" or (path and "kata" not in Path(path).name):
                raise ValueError("unsupported runtime registration")
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise KataRuntimeError(
                f"Docker runtime {self.runtime_name!r} must be registered for Kata "
                "on a Linux daemon; install Kata/QEMU first. No fallback to runc."
            ) from exc
        if resources is not None:
            if resources.cpu_limit and not info.get("CPUCfsQuota"):
                raise KataRuntimeError("The Docker host cannot enforce the requested CPU limit")
            if resources.mem_limit_bytes and not info.get("MemoryLimit"):
                raise KataRuntimeError("The Docker host cannot enforce the requested memory limit")

    @staticmethod
    def _entrypoint_args(run_args: list[str] | None) -> list[str]:
        args = list(run_args or [])
        if not args:
            return []
        if len(args) == 2 and args[0] == "--entrypoint":
            return args
        if len(args) == 1 and args[0].startswith("--entrypoint="):
            return args
        raise ValueError("Kata's offline profile accepts only --entrypoint in run_args")

    def acquire(
        self, *, image: str, command: list[str] | None = None,
        env: dict[str, str] | None = None, mounts: list[BindMount] | None = None,
        workspace_dir: str | None = None, platform: str | None = None,
        run_args: list[str] | None = None, resources: ContainerResources | None = None,
        acquire_timeout: float = 600.0,
    ) -> ContainerHandle:
        extra = self._entrypoint_args(run_args)
        if not image or image.startswith("-"):
            raise ValueError("Kata requires a nonempty image reference, not a Docker option")
        if any(not mount.read_only for mount in mounts or []):
            raise ValueError("Kata's offline profile only accepts read-only host input mounts")
        resource_args = _resource_run_flags(resources) if resources is not None else []
        self._check_runtime(resources)
        # Retain the name BEFORE docker run, so timeouts/partial creates can be
        # cleaned up even if Docker never returns a container ID.
        name = f"beagle-kata-{uuid.uuid4().hex}"
        handle = ContainerHandle(container_id=name, name=name)
        argv = self._docker_argv("run", "-d", "--name", name, "--runtime", self.runtime_name,
                                 "--network", "none")
        argv += resource_args
        if platform:
            argv += ["--platform", platform]
        if workspace_dir:
            argv += ["--workdir", workspace_dir]
        for mount in mounts or []:
            argv += ["-v", f"{mount.host_path.resolve()}:{mount.container_path}:ro"]
        process_env = os.environ.copy()
        for key, value in (env or {}).items():
            if not key or "=" in key:
                raise ValueError("Invalid container environment variable name")
            process_env[key] = value
            argv += ["--env", key]
        argv += [*extra, image, *(command or [])]
        try:
            result = subprocess.run(argv, capture_output=True, text=True,
                                    timeout=acquire_timeout, check=False, env=process_env)
            if result.returncode:
                raise KataRuntimeError(f"Kata container launch failed (exit {result.returncode}): {result.stderr.strip()}")
            container_id = result.stdout.strip()
            if not container_id:
                raise KataRuntimeError("Docker returned no container ID for the Kata launch")
            handle.container_id = container_id
            inspect = subprocess.run(self._docker_argv("inspect", container_id),
                                     capture_output=True, text=True, timeout=30, check=False)
            if inspect.returncode:
                raise KataRuntimeError("Cannot verify the launched Kata container")
            actual = json.loads(inspect.stdout)[0]
            if (actual["HostConfig"]["Runtime"] != self.runtime_name
                    or actual["HostConfig"]["NetworkMode"] != "none"):
                raise KataRuntimeError("Launched container does not match the requested Kata offline profile")
            return handle
        except BaseException as launch_error:
            try:
                self.destroy(handle)
            except KataRuntimeError as cleanup_error:
                raise KataRuntimeError(
                    f"Kata launch failed and cleanup could not be confirmed: {cleanup_error}",
                    handle=handle,
                ) from launch_error
            raise

    def destroy(self, handle: ContainerHandle) -> None:
        """Force-remove and confirm absence; retain the handle on any uncertainty.

        Docker owns VM teardown. A real-host smoke additionally checks that the
        associated QEMU process terminates; this API confirms Docker removal.
        """
        if not handle.container_id:
            return
        # A force remove also covers guests unable to respond to a graceful stop.
        try:
            subprocess.run(self._docker_argv("rm", "-f", handle.container_id),
                           capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass  # A lost removal response is resolved by an independent listing.
        try:
            result = subprocess.run(
                self._docker_argv("ps", "--all", "--quiet", "--no-trunc", "--filter",
                                 (f"name=^/{handle.name}$" if handle.container_id == handle.name
                                  else f"id={handle.container_id}")),
                capture_output=True, text=True, timeout=30, check=False,
            )
            if result.returncode or result.stdout.strip():
                raise KataRuntimeError(
                    f"Docker cannot confirm Kata container removal: {handle.container_id}", handle=handle)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise KataRuntimeError(
                f"Kata cleanup verification failed: {handle.container_id}", handle=handle) from exc
        handle.container_id = ""
