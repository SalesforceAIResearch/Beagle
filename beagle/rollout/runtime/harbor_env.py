"""``HarborEnvRuntime`` — adapt harbor's async ``BaseEnvironment`` to the sync
:class:`~beagle.rollout.runtime.protocol.ContainerRuntime` an agent expects.

This is the seam that keeps agent↔harness integration **M + N** instead of M × N:
an agent implements ``run(task, task_ctx, *, runtime)`` once, and each harness
supplies a ``ContainerRuntime`` over its native environment. Harbor owns the trial
container (it already provisioned it and will run the verifier + collect
artifacts), so here:

- ``acquire`` returns harbor's environment as the opaque handle — no new container;
- ``exec`` bridges each sync call to ``await environment.exec(...)`` on harbor's
  event loop (the agent runs in a worker thread, so this blocks that thread, not
  the loop);
- ``destroy`` is a no-op — harbor tears the container down in its trial cleanup.

Bind mounts are rejected: harbor's ``BaseEnvironment`` has no host-path bind hook
(it must also work on Modal/E2B), so agent source must arrive via ``git_clone``.

The harbor ``environment`` is duck-typed (``environment.exec(command, cwd, env,
timeout_sec, user) -> result`` with ``.stdout / .stderr / .return_code``) so this
module imports no harbor symbol and stays importable without harbor installed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import shlex
from typing import TYPE_CHECKING, Any

from beagle.rollout.runtime.runtime import ExecResult

if TYPE_CHECKING:
    from beagle.rollout.runtime.transport import BindMount

# Slack over the exec's own ``timeout_sec`` before we stop blocking on the future
# and give up on it — lets harbor's ``exec`` return its own timeout result first,
# so we only hit this backstop if the coroutine is genuinely wedged.
_RESULT_GRACE_SEC = 30.0
# Timeout / cancellation raised while awaiting the future. asyncio.CancelledError
# is a BaseException (not Exception), so it must be named explicitly.
_TIMEOUT_EXC = (concurrent.futures.TimeoutError, asyncio.TimeoutError, TimeoutError)
_CANCEL_EXC = (concurrent.futures.CancelledError, asyncio.CancelledError)


class HarborEnvRuntime:
    """A :class:`ContainerRuntime` backed by a single harbor trial environment.

    ``environment`` is harbor's ``BaseEnvironment`` for the current trial; ``loop``
    is the event loop it runs on (capture it inside the shim's async ``run`` via
    ``asyncio.get_running_loop()``). ``default_timeout`` bounds any exec whose caller
    passes no timeout.
    """

    def __init__(
        self,
        environment: Any,
        loop: asyncio.AbstractEventLoop,
        *,
        env_hook: Any | None = None,
    ) -> None:
        self._env = environment
        self._loop = loop
        # Optional per-command env transform, supplied by the HARNESS (not this generic runtime).
        # The pier shim passes ``environment.agent_process_env`` so a filtered-egress trial routes
        # the agent's commands through its Squid proxy; the harbor path passes nothing → unchanged.
        self._env_hook = env_hook

    def acquire(
        self,
        *,
        image: str = "",
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        mounts: list[BindMount] | None = None,
        workspace_dir: str | None = None,
        platform: str | None = None,
        run_args: list[str] | None = None,
        resources: Any | None = None,
        acquire_timeout: float | None = None,
    ) -> Any:
        """Return the harbor-provisioned container as the opaque handle.

        ``image`` / ``command`` / ``resources`` are ignored — harbor already
        started the trial container. ``mounts`` is rejected (no host bind-mount on
        ``BaseEnvironment``; use a ``git_clone`` transport). All other kwargs are
        accepted-and-ignored for ``ContainerRuntime`` substitutability.
        """
        if mounts:
            raise RuntimeError(
                "HarborEnvRuntime does not support bind mounts (harbor's "
                "BaseEnvironment has no host-path bind hook). Deliver agent source "
                "via a git_clone transport instead."
            )
        return self._env

    def exec(
        self,
        handle: Any,
        command: list[str],
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> ExecResult:
        """Run ``command`` (argv) in the trial container via harbor's async exec.

        argv is joined into the single shell string harbor's ``exec`` expects.
        ``user=None`` runs as harbor's ``default_user`` (the task's agent user, set
        by the trial runner). Per the runtime contract, a **timeout/cancel** maps to
        ``ExecResult(returncode=124, ...)`` and any other transport error to
        ``returncode=125`` (so the timeout-vs-crash signal survives) — never raises.
        """
        cmd = command if isinstance(command, str) else shlex.join(command)
        if self._env_hook is not None:   # harness-supplied env transform (pier: Squid egress proxy)
            env = self._env_hook(env)
        # A sub-1s timeout must NOT collapse to 0 → falsy → "no timeout"; round up to
        # at least 1s. ``None`` means the caller wants no timeout.
        timeout_sec = max(1, math.ceil(timeout)) if timeout is not None else None
        coro = self._env.exec(cmd, cwd=workdir, env=env or None, timeout_sec=timeout_sec, user=None)

        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        # Bound the wait so the worker thread can't wedge forever if the coroutine
        # doesn't honor ``timeout_sec`` (harbor cancels the *outer* run, not this
        # thread). ``None`` timeout → block (the caller opted out).
        deadline = (timeout_sec + _RESULT_GRACE_SEC) if timeout_sec is not None else None
        try:
            result = fut.result(timeout=deadline)
        except _TIMEOUT_EXC:
            fut.cancel()  # best-effort: stop the wedged coroutine
            return ExecResult(returncode=124, stdout="", stderr=f"exec exceeded {timeout_sec}s")
        except _CANCEL_EXC as e:
            fut.cancel()
            return ExecResult(returncode=124, stdout="", stderr=f"cancelled: {type(e).__name__}")
        except Exception as e:  # noqa: BLE001 — transport/backend error, distinct from timeout
            return ExecResult(returncode=125, stdout="", stderr=f"{type(e).__name__}: {e}")
        return ExecResult(
            returncode=int(getattr(result, "return_code", 1)),
            stdout=getattr(result, "stdout", None) or "",
            stderr=getattr(result, "stderr", None) or "",
        )

    def destroy(self, handle: Any) -> None:
        """No-op — harbor owns the trial container's lifecycle."""


__all__ = ["HarborEnvRuntime"]
