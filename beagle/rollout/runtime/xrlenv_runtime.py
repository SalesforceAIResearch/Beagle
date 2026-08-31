"""XrlenvDockerRuntime — cluster-mode runtime via ``xrlenv.from_env()``.

Drop-in sibling to :class:`LocalDockerRuntime`. Satisfies the same
``acquire`` / ``exec`` / ``destroy`` shape, but every operation is routed
through xrlenv's cluster scheduler: image-affinity, capacity accounting,
and cancellation are applied transparently.

xrlenv (and its transitive ``docker`` SDK) are imported lazily — strictly
inside method bodies. Users who never initialise ``vendor/xrlenv`` see zero
import errors when importing or instantiating other parts of this package;
xrlenv is only loaded when ``XrlenvDockerRuntime(...)`` is constructed.

The connection config can also be set via environment (``XRLENV_GRPC_HOST``
/ ``XRLENV_GRPC_PORT`` / ``XRLENV_CONSUMER_TOKEN`` / ``XRLENV_GRPC_SECURE``)
— passing no kwargs then yields cluster mode if the host is set, or
LocalDocker mode if it isn't. Matches ``xrlenv.from_env()`` exactly.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import uuid
from collections.abc import Iterator
from typing import Any

from beagle.rollout.runtime.runtime import (
    ContainerHandle,
    ContainerResources,
    ExecResult,
)
from beagle.rollout.runtime.transport import BindMount

LOGGER = logging.getLogger(__name__)
_CONTAINER_NAME_PREFIX = "beagle-"


def _already_gone(exc: BaseException) -> bool:
    """True when a destroy raced a teardown that already removed the container.

    A per-task ``destroy`` running as a Ctrl-C group teardown (``terminate_raw_group``) tears the
    same containers down hits a cluster ``… session/rollout not found. Acquire first.`` — the
    container is already gone, which is exactly what destroy wanted. Treat that as benign (return
    quietly) instead of logging a scary ``may leak`` traceback."""
    return "not found" in str(exc).lower()

# Per-acquire pull/acquire deadline forwarded to xrlenv as the
# ``xrlenv.acquire_timeout_s`` reserved label. Benchmarks with a distinct
# multi-GB image per instance make every acquire a cold pull; the xrlenv
# server default (600 s) can be too tight on a contended cluster. 1800 s
# gives a cold pull room to land.
_ACQUIRE_TIMEOUT_S = 1800.0

# Per-task labels (e.g. xrlenv.task_key, xrlenv.group_id) flow into
# acquire() via this contextvar so adapters can inject task identity
# without changing the runtime acquire() Protocol surface (shared with
# LocalDockerRuntime). See acquire_labels().
_ACQUIRE_LABELS: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "xrlenv_acquire_labels", default={},
)


@contextlib.contextmanager
def acquire_labels(labels: dict[str, str]) -> Iterator[None]:
    """While active, :meth:`XrlenvDockerRuntime.acquire` merges ``labels``
    into the docker labels passed to ``containers.run(...)``.

    Lets adapters inject per-task identity labels — ``xrlenv.task_key``,
    ``xrlenv.group_id`` — without touching the shared runtime ``acquire()``
    signature::

        with acquire_labels({"xrlenv.task_key": task_id,
                             "xrlenv.group_id": run_id}):
            handle = rt.acquire(image=..., command=...)

    The xrlenv cluster path reads these labels server-side; the scheduler
    also consumes ``xrlenv.task_key`` for anti-affinity.
    """
    token = _ACQUIRE_LABELS.set(dict(labels))
    try:
        yield
    finally:
        _ACQUIRE_LABELS.reset(token)


# Sentinel messages explaining why a kwarg was refused (grep-able in logs).
_REFUSE_BIND_MOUNT = (
    "XrlenvDockerRuntime does not propagate BindMount through xrlenv's "
    "cluster path — the host filesystem is not reachable from a remote "
    "node. Switch the agent source to GitClone (transport.py) for cluster "
    "runs, or use `runtime: local` to keep the bind-mount path."
)
_REFUSE_WORKSPACE_DIR = (
    "XrlenvDockerRuntime does not propagate workspace_dir through xrlenv's "
    "cluster path. Set the workdir inside the image, or use `runtime: local`."
)
_REFUSE_PLATFORM = (
    "XrlenvDockerRuntime does not propagate platform through xrlenv's "
    "cluster path. Build images for the cluster's target arch, or use "
    "`runtime: local`."
)
_REFUSE_DISK = (
    "XrlenvDockerRuntime does not propagate ContainerResources."
    "disk_limit_bytes — xrlenv's acquire_container does not wire a "
    "writable-disk cap today. Drop the field, or use `runtime: local`."
)
_REFUSE_GPUS = (
    "XrlenvDockerRuntime does not propagate ContainerResources.gpus — "
    "xrlenv's acquire_container does not wire GPU requests today. Drop "
    "the field, or use `runtime: local`."
)


class XrlenvDockerRuntime:
    """Cluster-mode runtime backed by ``xrlenv.from_env()``.

    Constructor kwargs mirror xrlenv's ``from_env`` connect-mode form. All
    are optional; when unset, xrlenv reads ``XRLENV_GRPC_HOST`` etc. from
    the environment. With nothing set anywhere the client falls back to
    LocalDocker mode against the host daemon.
    """

    def __init__(
        self,
        *,
        grpc_host: str | None = None,
        grpc_port: int | None = None,
        consumer_token: str | None = None,
        grpc_secure: bool | None = None,
        run_id: str | None = None,
    ) -> None:
        # Lazy import: ``vendor/xrlenv`` (and its docker-py dep) are only
        # required for the cluster path; the import only fires here.
        import xrlenv  # noqa: PLC0415

        self._client = xrlenv.from_env(
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            consumer_token=consumer_token,
            grpc_secure=grpc_secure,
        )
        # Stamped as ``xrlenv.group_id`` on every container's labels so the
        # admin /rollouts view groups all tasks of one run together.
        self._run_id = run_id

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
        acquire_timeout: float | None = None,
    ) -> ContainerHandle:
        """Start a container in the background. Mirrors
        :meth:`LocalDockerRuntime.acquire` but fails loudly on kwargs the
        xrlenv cluster path does not currently propagate.

        ``acquire_timeout`` is accepted for :class:`ContainerRuntime`
        substitutability but ignored: xrlenv applies its own pull/acquire
        deadline via the ``xrlenv.acquire_timeout_s`` reserved label below.

        Supported in cluster mode: ``image``, ``command``, ``env``,
        ``run_args`` (``--entrypoint`` only), ``resources`` (``cpu_limit`` /
        ``mem_limit_bytes`` -> ``nano_cpus`` / ``mem_limit``). Refused with
        a clear ``RuntimeError``: non-empty ``mounts``, non-None
        ``workspace_dir`` / ``platform``, and ``disk_limit_bytes`` / ``gpus``.
        """
        if mounts:
            raise RuntimeError(_REFUSE_BIND_MOUNT)
        if workspace_dir is not None:
            raise RuntimeError(_REFUSE_WORKSPACE_DIR)
        if platform is not None:
            raise RuntimeError(_REFUSE_PLATFORM)

        name = f"{_CONTAINER_NAME_PREFIX}{uuid.uuid4().hex[:8]}"

        kwargs: dict[str, Any] = {}
        if run_args:
            kwargs.update(_translate_run_args(run_args))

        # Forward the benchmark's per-task resource cap. xrlenv's drop-in
        # ``containers.run`` accepts docker-py's ``nano_cpus`` / ``mem_limit``
        # and hoists them into the ``acquire_container`` ResourceSpec, so the
        # control plane places + cgroup-isolates against the real footprint.
        # disk_limit_bytes / gpus are not wired through xrlenv yet.
        if resources is not None:
            if resources.disk_limit_bytes is not None:
                raise RuntimeError(_REFUSE_DISK)
            if resources.gpus:
                raise RuntimeError(_REFUSE_GPUS)
            if resources.cpu_limit is not None and resources.cpu_limit > 0:
                kwargs["nano_cpus"] = int(resources.cpu_limit * 1_000_000_000)
            if (
                resources.mem_limit_bytes is not None
                and resources.mem_limit_bytes > 0
            ):
                kwargs["mem_limit"] = int(resources.mem_limit_bytes)

        # Merge run-level (``xrlenv.group_id``) + per-task (``acquire_labels``)
        # labels; cluster mode forwards them onto the ``RawRolloutRecord``.
        labels: dict[str, str] = {}
        if self._run_id:
            labels["xrlenv.group_id"] = self._run_id
        ctx_labels = _ACQUIRE_LABELS.get()
        if ctx_labels:
            labels.update(ctx_labels)
        # Widen the xrlenv pull/acquire deadline. Passed as a reserved label,
        # NOT a kwarg: ``containers.run`` rejects unknown kwargs but labels
        # pass through; the xrlenv drop-in hoists ``xrlenv.acquire_timeout_s``.
        labels["xrlenv.acquire_timeout_s"] = str(_ACQUIRE_TIMEOUT_S)

        try:
            container = self._client.containers.run(
                image=image,
                command=command,
                name=name,
                detach=True,
                environment=env or None,
                labels=labels or None,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"xrlenv containers.run failed for image={image!r} "
                f"name={name!r}: {e}"
            ) from e

        return ContainerHandle(container_id=container.id, name=name)

    def exec(
        self,
        handle: ContainerHandle,
        command: list[str],
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> ExecResult:
        """Run ``command`` in the container via docker-py ``exec_run``.

        Wraps the argv with coreutils ``timeout`` so timeouts surface as
        ``rc=124`` — matching :meth:`LocalDockerRuntime.exec`. ``timeout=None``
        and ``timeout<=0`` both mean "no timeout".
        """
        if not handle.container_id:
            raise RuntimeError("cannot exec on a destroyed/empty handle")

        if timeout is not None and timeout > 0:
            # ``-k 10`` is the POSIX-style short form of ``--kill-after=10``;
            # GNU coreutils and BusyBox both accept it (BusyBox rejects the
            # long form), and several benchmark images are Alpine/BusyBox.
            command = ["timeout", "-k", "10", str(int(timeout)), *command]

        try:
            container = self._client.containers.get(handle.container_id)
            result = container.exec_run(
                cmd=command,
                environment=env or None,
                workdir=workdir,
                demux=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"xrlenv exec_run failed on container={handle.container_id!r}: {e}"
            ) from e

        # ``demux=True`` returns (stdout_bytes, stderr_bytes); either element
        # may be None when its stream was empty.
        stdout_b, stderr_b = result.output if result.output else (None, None)
        return ExecResult(
            returncode=result.exit_code if result.exit_code is not None else -1,
            stdout=(stdout_b or b"").decode(errors="replace"),
            stderr=(stderr_b or b"").decode(errors="replace"),
        )

    def destroy(self, handle: ContainerHandle) -> None:
        """Stop + remove the container. Idempotent; errors are swallowed
        with a logged warning."""
        if not handle.container_id:
            return
        cid = handle.container_id
        handle.container_id = ""

        import docker.errors  # noqa: PLC0415

        try:
            container = self._client.containers.get(cid)
        except docker.errors.NotFound:
            return
        except Exception:
            LOGGER.warning("destroy: containers.get(%s) failed; nothing to clean", cid, exc_info=True)
            return

        try:
            container.stop(timeout=30)
        except docker.errors.NotFound:
            return
        except Exception as exc:
            if _already_gone(exc):
                return  # torn down under us (e.g. a Ctrl-C group teardown) — nothing to clean
            LOGGER.warning("destroy: stop(%s) failed; continuing to remove", cid, exc_info=True)

        try:
            container.remove(force=True)
        except docker.errors.NotFound:
            return
        except Exception as exc:
            if _already_gone(exc):
                return  # already gone — not a leak, so don't alarm with a traceback
            LOGGER.warning("destroy: remove(%s) failed; container may leak", cid, exc_info=True)

    def stop_run(self, run_id: str) -> Any:
        """Actively destroy every container of this run (those tagged ``xrlenv.group_id ==
        run_id``) on the cluster — a node-confirmed teardown that frees capacity NOW instead of
        leaving the containers for xrlenv's raw-liveness reaper (~120 s). Best-effort; wired to
        the CLI's Ctrl-C handler (see :func:`beagle.rollout.interrupt.stop_run_on_sigint`).

        Covers the containers THIS runtime acquired (the agent containers, which carry the
        group label from :meth:`acquire`). Grader / harbor containers created via other xrlenv
        clients aren't tagged with the run's group and still fall to the reaper.

        Returns the terminate report (``None`` in LocalDocker / caller-managed mode, or when
        there's no ``run_id`` to scope by)."""
        if not run_id:
            return None
        return self._client.terminate_raw_group(run_id)

    @contextlib.contextmanager
    def rollout_scope(
        self,
        *,
        displayed_name: str,
        artifact_path: str | None = None,
    ) -> Iterator[None]:
        """Per-task telemetry scope — wraps the agent's task body so the
        admin ``/rollouts/raw`` view shows ``displayed_name`` and links to
        ``artifact_path``.

        Optional method (off the Protocol surface): adapters call it via
        ``getattr(runtime, "rollout_scope", contextlib.nullcontext)`` so
        local runtimes incur zero overhead.
        """
        import xrlenv  # noqa: PLC0415

        with xrlenv.rollout_metadata(
            displayed_name=displayed_name,
            artifact_path=artifact_path,
        ):
            yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _translate_run_args(run_args: list[str]) -> dict[str, Any]:
    """Translate ``acquire(run_args=...)`` flags into docker-py
    ``containers.run`` kwargs.

    Recognised today: ``--entrypoint <value>`` -> ``entrypoint=value`` (an
    empty value clears the image ENTRYPOINT so ``sleep infinity`` can run as
    PID 1). Unknown flags raise ``NotImplementedError`` — silent drops would
    surface as a benchmark misconfiguration much later.
    """
    out: dict[str, Any] = {}
    i = 0
    while i < len(run_args):
        flag = run_args[i]
        if flag == "--entrypoint":
            if i + 1 >= len(run_args):
                raise ValueError(f"--entrypoint missing value in {run_args!r}")
            value = run_args[i + 1]
            out["entrypoint"] = value if value else [""]
            i += 2
        else:
            raise NotImplementedError(
                f"XrlenvDockerRuntime: run_args flag {flag!r} not yet "
                f"translated to docker-py kwargs. Add a case to "
                f"_translate_run_args(). Full args: {run_args!r}"
            )
    return out


__all__ = ["XrlenvDockerRuntime", "acquire_labels"]
