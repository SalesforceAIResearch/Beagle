"""Stub HTTP-over-uds server (spec 01).

The endpoint surface is a strict subset of E2B's REST API plus the XRLEnv-
specific ``/env/{setup,step,teardown}`` triplet that the in-sandbox EnvAdapter
serves (spec 14). aiohttp is the only runtime dep — chosen because it ships
``UnixSite``/``UnixConnector`` out of the box, supports HTTP/1.1 chunked
responses, and is widely pre-installed in the slim Python images we build on.

Endpoints (Slice 1):
    GET  /healthz                — liveness probe
    POST /commands               — exec a one-shot command
    GET  /files{path}            — read a small file
    POST /files{path}            — write a small file (body is the bytes)
    POST /env/setup              — load the EnvAdapter and call .setup()
    POST /env/step               — call adapter.step(action)
    POST /env/teardown           — call adapter.teardown()
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
from pathlib import Path
from typing import Any

from aiohttp import web

from xrlenv.envs.base import EnvAdapter

LOGGER = logging.getLogger(__name__)


class StubState:
    """Mutable runtime state of the stub.

    The stub is single-instance per sandbox so we don't need locking around the
    adapter — at most one ``/env/step`` call is in flight at a time per the
    spec-02 lifecycle.
    """

    adapter: EnvAdapter | None = None
    adapter_module: str | None = None
    adapter_class: str | None = None


def build_app() -> web.Application:
    state = StubState()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["state"] = state

    app.router.add_get("/healthz", _healthz)
    app.router.add_post("/commands", _commands)
    app.router.add_get("/files{path:.*}", _read_file)
    app.router.add_post("/files{path:.*}", _write_file)
    app.router.add_post("/env/setup", _env_setup)
    app.router.add_post("/env/step", _env_step)
    app.router.add_post("/env/teardown", _env_teardown)
    return app


# ──────────────────────────────────────────────────────────────────────────────
# Handlers
# ──────────────────────────────────────────────────────────────────────────────


async def _healthz(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def _commands(request: web.Request) -> web.Response:
    """Run a one-shot command; return stdout/stderr/exit_code as JSON.

    Streaming exec lands in Slice 2 (spec 01); for Slice 1 we collect the
    output and return it in a single response.
    """
    body = await request.json()
    cmd = body.get("cmd")
    if not isinstance(cmd, list) or not all(isinstance(p, str) for p in cmd):
        raise web.HTTPBadRequest(reason="cmd must be a list[str]")
    timeout_s = float(body.get("timeout_s") or 30.0)
    env_extra: dict[str, str] = body.get("env") or {}
    # Default ``cwd=None`` so the subprocess inherits the container's
    # WORKDIR (matches ``docker exec`` semantics). Forcing /sandbox
    # here breaks any benchmark image whose WORKDIR is something else
    # (tb2's fix-git uses /app); FileNotFoundError on /sandbox would
    # propagate to aiohttp as a 500 and surface as
    # ``StubResponseError`` at the platform — mysteriously, with no
    # connection to "the wrong cwd default."
    cwd = body.get("cwd") or None

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env={**os.environ, **env_extra},
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        exit_code = proc.returncode if proc.returncode is not None else -1
        timed_out = False
    except TimeoutError:
        proc.kill()
        await proc.wait()
        stdout, stderr = b"", b""
        exit_code = 124
        timed_out = True

    return web.json_response(
        {
            "exit_code": exit_code,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "timed_out": timed_out,
        }
    )


async def _read_file(request: web.Request) -> web.Response:
    path = "/" + request.match_info["path"]
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(reason=str(exc)) from exc
    return web.Response(body=data, content_type="application/octet-stream")


async def _write_file(request: web.Request) -> web.Response:
    path = Path("/" + request.match_info["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    data = await request.read()
    path.write_bytes(data)
    return web.json_response({"path": str(path), "size": len(data)})


async def _env_setup(request: web.Request) -> web.Response:
    state: StubState = request.app["state"]
    body = await request.json()

    module = body.get("adapter_module")
    classname = body.get("adapter_class")
    init_params = body.get("init_params") or {}
    sandbox_id = body.get("sandbox_id") or "stub"
    if not module or not classname:
        raise web.HTTPBadRequest(
            reason="env/setup requires adapter_module and adapter_class"
        )

    try:
        adapter = _instantiate_adapter(module, classname, sandbox_id=sandbox_id)
    except Exception as exc:
        LOGGER.exception("adapter import failed")
        return web.json_response(
            {"error": "adapter_import_failed", "message": str(exc)}, status=500
        )

    state.adapter = adapter
    state.adapter_module = module
    state.adapter_class = classname

    # Log + return a traceback-bearing 500 if ``adapter.setup`` raises.
    # Without this, the consumer's stub_client just sees aiohttp's
    # generic 500 and loses the per-rollout diagnostic trail — e.g.
    # an EnvAdapter that fails on ``os.makedirs(/sandbox)`` because
    # the upstream image runs as a non-root user with no /sandbox
    # writable. Surfacing the traceback here turns "smoke failed"
    # into "smoke failed because <X>".
    try:
        obs = await adapter.setup(init_params)
    except Exception as exc:
        LOGGER.exception(
            "adapter.setup failed (module=%s class=%s sandbox=%s)",
            module, classname, sandbox_id,
        )
        return web.json_response(
            {
                "error": "adapter_setup_failed",
                "message": f"{type(exc).__name__}: {exc}",
            },
            status=500,
        )
    caps = type(adapter).capabilities()
    return web.json_response(
        {
            "obs": obs,
            "capabilities": {
                "xrlenv_api_version_supported": list(caps.xrlenv_api_version_supported),
                "supported_reward_modes": sorted(caps.supported_reward_modes),
            },
        }
    )


async def _env_step(request: web.Request) -> web.Response:
    state: StubState = request.app["state"]
    if state.adapter is None:
        raise web.HTTPBadRequest(reason="env/step called before env/setup")
    body = await request.json()
    action = body.get("action")
    result = await state.adapter.step(action)
    return web.json_response(
        {
            "obs": result.obs,
            "reward": result.reward,
            "done": result.done,
            "truncated": result.truncated,
            "info": result.info,
        }
    )


async def _env_teardown(request: web.Request) -> web.Response:
    state: StubState = request.app["state"]
    if state.adapter is None:
        # Idempotent — teardown without setup is a no-op.
        return web.json_response({"status": "noop"})
    await state.adapter.teardown()
    state.adapter = None
    return web.json_response({"status": "ok"})


# ──────────────────────────────────────────────────────────────────────────────
# Adapter loader
# ──────────────────────────────────────────────────────────────────────────────


def _instantiate_adapter(module: str, classname: str, *, sandbox_id: str) -> EnvAdapter:
    mod = importlib.import_module(module)
    cls = getattr(mod, classname)
    if not isinstance(cls, type) or not issubclass(cls, EnvAdapter):
        raise TypeError(f"{module}.{classname} is not an EnvAdapter subclass")
    # SyncEnvAdapter takes ``sandbox_id`` kwarg; plain EnvAdapter takes no args.
    try:
        return cls(sandbox_id=sandbox_id)  # type: ignore[call-arg]
    except TypeError:
        return cls()


# ──────────────────────────────────────────────────────────────────────────────
# Server lifecycle
# ──────────────────────────────────────────────────────────────────────────────


class StubServer:
    """aiohttp app bound to a uds (Linux) or a TCP port (macOS fallback).

    Boundary owner of the uds path: the server creates the socket on
    ``serve_forever``, removes any stale socket inherited from a crashed
    previous instance, and unlinks on shutdown.
    """

    def __init__(
        self,
        *,
        uds_path: str | None = None,
        bind_host: str | None = None,
        bind_port: int | None = None,
    ) -> None:
        if not uds_path and bind_port is None:
            raise ValueError("StubServer requires either uds_path or bind_port")
        if uds_path and bind_port is not None:
            raise ValueError("StubServer accepts uds_path *or* bind_port, not both")
        self._uds_path = uds_path
        self._bind_host = bind_host or "0.0.0.0"
        self._bind_port = bind_port
        self._app = build_app()

    async def serve_forever(self) -> None:
        runner = web.AppRunner(self._app)
        await runner.setup()
        site: web.BaseSite
        if self._uds_path is not None:
            with _silently_unlink():
                os.unlink(self._uds_path)
            site = web.UnixSite(runner, self._uds_path)
        else:
            assert self._bind_port is not None
            site = web.TCPSite(runner, host=self._bind_host, port=self._bind_port)

        await site.start()
        if self._uds_path is not None:
            # Open the socket up for cross-uid connect(). The stub runs as
            # the Dockerfile's USER sandbox (uid 1000); the host node-agent
            # runs as the xrlenv system user (uid ~990 on AL2023). Default
            # umask leaves the bound socket at 0o644 (owner rw, others r),
            # which means the host process gets 'Permission denied' on
            # connect() because connect requires write on the socket file.
            # 0o666 lets any uid connect — safe because the socket lives in
            # the per-sandbox uuid4-named host dir, gated by the parent
            # runs_root's 0o755 mode.
            os.chmod(self._uds_path, 0o666)
            LOGGER.info("stub listening on uds=%s", self._uds_path)
        else:
            LOGGER.info("stub listening on tcp=%s:%d", self._bind_host, self._bind_port)

        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()
            if self._uds_path is not None:
                with _silently_unlink():
                    os.unlink(self._uds_path)


def _silently_unlink() -> Any:
    from contextlib import suppress

    return suppress(FileNotFoundError, IsADirectoryError, PermissionError)
