"""LocalDockerRuntime + supporting data types.

Thin sync wrapper around the local `docker` CLI. Each method shells
out via subprocess — no daemon connection, no Python SDK.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import uuid
from dataclasses import dataclass

from beagle.rollout.runtime.transport import BindMount

DOCKER = "docker"
_CONTAINER_NAME_PREFIX = "beagle-"
# Default ceiling for `docker run -d`. Big enough to absorb a cold-cache
# pull of a multi-GB benchmark image without making a stuck pull hang the
# run for an hour. Callers that pre-pull effectively never need more than
# a couple of seconds here, so the default rarely bites.
_DEFAULT_ACQUIRE_TIMEOUT_SEC = 600.0


@dataclass(frozen=True)
class ExecResult:
    """Result of one `exec()` call. Plain values, no surprises."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class ContainerHandle:
    """Reference returned by `acquire()`. Adapters pass it back to
    subsequent calls (`exec`, `destroy`, etc.) and don't need to read
    the fields. The local runtime clears `container_id` only after
    confirming removal, retaining it on failure so cleanup can be retried.
    """

    container_id: str
    name: str


class ContainerCleanupError(RuntimeError):
    """Local Docker cleanup could not confirm removal of a container."""

    def __init__(self, container_id: str, detail: str) -> None:
        self.container_id = container_id
        super().__init__(
            f"Container {container_id} removal could not be confirmed: {detail}. "
            "The container may still exist; its handle is retained for cleanup retry."
        )


@dataclass(frozen=True)
class ContainerResources:
    """A benchmark's declared per-task resource cap.

    Adapters populate this from wherever their benchmark declares
    resources. Every field is optional: ``None`` means "the benchmark
    says nothing about this dimension" and the runtime leaves it uncapped.

    Runtimes honor whatever they are handed; the decision of *whether* to
    pass a cap, and what value, belongs to the adapter — not here. New
    resource dimensions are added as fields here, so the ``acquire()``
    signature across the runtimes stays stable.
    """

    cpu_limit: float | None = None        # hard CPU cap, in cores
    mem_limit_bytes: int | None = None    # hard memory cap, in bytes
    disk_limit_bytes: int | None = None   # hard writable-disk cap, in bytes
    gpus: int | None = None               # GPU count (0 / None == no GPU)


def _resource_run_flags(r: ContainerResources) -> list[str]:
    """Render :class:`ContainerResources` into ``docker run`` flags.

    Raises ``NotImplementedError`` for dimensions not yet wired in the
    local runtime (``disk_limit_bytes``, ``gpus``) — failing loud beats
    silently dropping a cap the benchmark asked for.
    """
    flags: list[str] = []
    if r.cpu_limit is not None and r.cpu_limit > 0:
        flags += ["--cpus", str(r.cpu_limit)]
    if r.mem_limit_bytes is not None and r.mem_limit_bytes > 0:
        flags += ["--memory", str(r.mem_limit_bytes)]
    if r.disk_limit_bytes is not None:
        raise NotImplementedError(
            "ContainerResources.disk_limit_bytes is not wired in "
            "LocalDockerRuntime yet (docker --storage-opt size is "
            "storage-driver-dependent)."
        )
    if r.gpus:
        raise NotImplementedError(
            "ContainerResources.gpus is not wired in LocalDockerRuntime yet."
        )
    return flags


class LocalDockerRuntime:
    """Sync wrapper around the local `docker` CLI.

    Each method shells out via subprocess. Lifecycle::

        rt = LocalDockerRuntime()
        h = rt.acquire(image="alpine:3", command=["sleep", "infinity"])
        try:
            r = rt.exec(h, ["echo", "hi"])
            assert r.ok and r.stdout.strip() == "hi"
        finally:
            rt.destroy(h)

    `acquire(...)` does NOT pass `--rm` — we destroy explicitly so the
    container stays around long enough for follow-up exec/log calls.
    """

    def acquire(
        self,
        *,
        image: str,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        mounts: list[BindMount] | None = None,
        workspace_dir: str | None = None,
        platform: str | None = None,
        run_args: list[str] | None = None,
        resources: ContainerResources | None = None,
        acquire_timeout: float = _DEFAULT_ACQUIRE_TIMEOUT_SEC,
    ) -> ContainerHandle:
        """Start a container in the background. Returns a handle.

        ``cpu_limit`` / ``mem_limit_bytes`` render as ``--cpus`` /
        ``--memory``; a field left ``None`` leaves that dimension uncapped.

        Raises:
            RuntimeError: docker CLI missing or ``docker run`` returned non-zero.
        """
        if shutil.which(DOCKER) is None:
            raise RuntimeError(f"{DOCKER!r} CLI not on PATH")

        name = f"{_CONTAINER_NAME_PREFIX}{uuid.uuid4().hex[:8]}"
        argv: list[str] = [DOCKER, "run", "-d", "--name", name]

        if platform:
            argv += ["--platform", platform]
        if resources is not None:
            argv += _resource_run_flags(resources)
        if workspace_dir:
            argv += ["-w", workspace_dir]
        for k, v in (env or {}).items():
            argv += ["-e", f"{k}={v}"]
        for m in (mounts or []):
            host = str(m.host_path.resolve())
            mode = ":ro" if m.read_only else ""
            argv += ["-v", f"{host}:{m.container_path}{mode}"]
        argv += list(run_args or [])
        argv.append(image)
        if command is not None:
            argv += list(command)
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=acquire_timeout, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"docker run failed (rc={result.returncode}): "
                f"stderr={result.stderr.strip()!r}\n"
                f"command was: {shlex.join(argv)}"
            )
        return ContainerHandle(container_id=result.stdout.strip(), name=name)

    def exec(
        self,
        handle: ContainerHandle,
        command: list[str],
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> ExecResult:
        """Run ``command`` in a running container via ``docker exec``.

        On timeout the exec process is killed and an ExecResult with
        ``returncode=124`` is returned (matches ``timeout(1)``).
        ``None`` means no timeout.

        Raises:
            RuntimeError: handle has no container_id (already destroyed).
        """
        if not handle.container_id:
            raise RuntimeError("cannot exec on a destroyed/empty handle")

        argv: list[str] = [DOCKER, "exec"]
        if workdir:
            argv += ["-w", workdir]
        process_env = None
        if env:
            # `docker exec -e NAME=value` exposes credentials in the host process argv (`ps`) and
            # command diagnostics. Put values in the docker CLI's environment and ask Docker to
            # copy each named variable instead; argv contains names only.
            process_env = os.environ.copy()
            process_env.update(env)
            for k in env:
                argv += ["-e", k]
        argv.append(handle.container_id)
        argv += list(command)
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True,
                timeout=timeout, check=False, env=process_env,
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                returncode=124,
                stdout=e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
                stderr=e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or ""),
            )
        return ExecResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def destroy(self, handle: ContainerHandle) -> None:
        """Stop, force-remove, and verify removal through the Docker daemon.

        A failed/timed-out stop must not prevent force-removal. Only a
        successful listing that excludes the container confirms cleanup,
        including when another actor removed it or a removal response was lost.
        Confirmation clears the handle; subsequent calls are no-ops.

        Raises:
            ContainerCleanupError: the container remains or its absence cannot
                be verified. The handle retains its ID so callers can retry.
        """
        if not handle.container_id:
            return
        cid = handle.container_id
        failures: list[str] = []
        for args in (["stop", cid], ["rm", "-f", cid]):
            try:
                result = subprocess.run(
                    [DOCKER, *args], capture_output=True, text=True, timeout=30, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures.append(f"{args[0]} failed: {exc}")
            else:
                if result.returncode != 0:
                    failures.append(
                        f"{args[0]} returned {result.returncode}: {result.stderr.strip()}"
                    )

        # Use a successful, filtered listing rather than interpreting an inspect
        # error as absence: daemon outages and permission errors must fail closed.
        # --all includes stopped containers, whose removal also needs confirmation.
        try:
            remaining = subprocess.run(
                [DOCKER, "ps", "--all", "--quiet", "--no-trunc", "--filter", f"id={cid}"],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"verification failed: {exc}")
            raise ContainerCleanupError(cid, "; ".join(failures)) from exc

        if remaining.returncode != 0:
            failures.append(
                f"verification returned {remaining.returncode}: {remaining.stderr.strip()}"
            )
        elif remaining.stdout.strip():
            failures.append("Docker still lists the container")
        else:
            handle.container_id = ""
            return

        raise ContainerCleanupError(cid, "; ".join(failures))


__all__ = [
    "ContainerCleanupError",
    "ContainerHandle",
    "ContainerResources",
    "ExecResult",
    "LocalDockerRuntime",
]
