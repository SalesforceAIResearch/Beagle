"""HTTP client used by the node agent to talk to the in-sandbox stub.

aiohttp's connectors give us a single HTTP/1.1 stack across both transports:
``UnixConnector`` for the Linux ``unix://`` endpoint and ``TCPConnector`` for
the macOS-Docker-Desktop ``tcp://`` fallback. The base URL is irrelevant for
``UnixConnector`` (the connector ignores host) but ``TCPConnector`` uses the
real host:port baked into the endpoint URI.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, suppress
from typing import Any
from urllib.parse import urlparse

import aiohttp

from xrlenv.errors import XRLEnvError


class StubResponseError(XRLEnvError):
    """Raised by :class:`StubClient` when the in-sandbox stub returns a
    non-2xx response. Carries the stub's structured ``{error, message}``
    payload so the coordinator / smoke log surfaces the actual failure
    cause instead of a bare HTTP status (audit follow-up: an EnvAdapter
    that fails inside ``setup`` used to surface as just "500 Internal
    Server Error" with no traceback link).
    """

    def __init__(
        self, message: str, *, status: int, error: str, message_text: str = "",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error = error
        self.message_text = message_text


#: Default per-request HTTP timeout for stub calls. The platform's
#: real upper bounds are the rollout's hard_s (deadline-watcher) and
#: the adapter's per-step subprocess timeout (from
#: init_params['step_timeout_s'] which the resolver populates from
#: ``[agent].timeout_sec`` in task.toml). The HTTP-level cap here is
#: a safety net for the case where the stub silently hangs without
#: responding — it should be larger than any realistic step.
#:
#: 1 hour is generous enough for tb2's longest tasks (dna-insert at
#: 1800s, crack-7z-hash brute-forcing a 7-zip password) and still
#: bounds an absent-stub scenario.
_DEFAULT_REQUEST_TIMEOUT_S: float = 3600.0


class StubClient:
    """One client per sandbox; share a single ClientSession for all calls."""

    def __init__(
        self,
        endpoint: str,
        *,
        request_timeout_s: float = _DEFAULT_REQUEST_TIMEOUT_S,
    ) -> None:
        self._endpoint = endpoint
        self._kind, self._uds_path, self._base_url = _parse_endpoint(endpoint)
        self._timeout_s = request_timeout_s
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> StubClient:
        self._session = self._build_session()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    @asynccontextmanager
    async def _session_ctx(self) -> Any:
        if self._session is None:
            session = self._build_session()
            try:
                yield session
            finally:
                await session.close()
        else:
            yield self._session

    def _build_session(self) -> aiohttp.ClientSession:
        timeout = aiohttp.ClientTimeout(total=self._timeout_s)
        if self._kind == "uds":
            assert self._uds_path is not None
            connector: aiohttp.BaseConnector = aiohttp.UnixConnector(path=self._uds_path)
        else:
            connector = aiohttp.TCPConnector()
        return aiohttp.ClientSession(connector=connector, timeout=timeout)

    # ── Endpoints ────────────────────────────────────────────────────────────

    async def healthz(self) -> dict[str, Any]:
        return await self._get_json("/healthz")

    async def env_setup(
        self,
        adapter_module: str,
        adapter_class: str,
        init_params: dict[str, Any],
        sandbox_id: str,
        *,
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return await self._post_json(
            "/env/setup",
            {
                "adapter_module": adapter_module,
                "adapter_class": adapter_class,
                "init_params": init_params,
                "sandbox_id": sandbox_id,
            },
            request_timeout_s=request_timeout_s,
        )

    async def env_step(
        self, action: Any, *, request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return await self._post_json(
            "/env/step", {"action": action},
            request_timeout_s=request_timeout_s,
        )

    async def env_teardown(
        self, *, request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return await self._post_json(
            "/env/teardown", {},
            request_timeout_s=request_timeout_s,
        )

    async def commands(
        self,
        cmd: list[str],
        *,
        timeout_s: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._post_json(
            "/commands",
            {"cmd": cmd, "timeout_s": timeout_s, "cwd": cwd, "env": env},
        )

    # ── HTTP plumbing ────────────────────────────────────────────────────────

    async def _get_json(self, path: str) -> dict[str, Any]:
        async with (
            self._session_ctx() as session,
            session.get(self._base_url + path) as resp,
        ):
            resp.raise_for_status()
            data = await resp.json()
            return data if isinstance(data, dict) else {"value": data}

    async def _post_json(
        self,
        path: str,
        body: dict[str, Any],
        *,
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        # D17 stage 2: per-call cap overrides the session-level
        # ClientTimeout. Aiohttp accepts a per-request ``timeout=``
        # kwarg on ``session.post`` that supersedes the session
        # default for that one call. ``None`` (or non-positive) means
        # "no per-call override; use the session default."
        post_kwargs: dict[str, Any] = {"json": body}
        if request_timeout_s is not None and request_timeout_s > 0:
            post_kwargs["timeout"] = aiohttp.ClientTimeout(total=request_timeout_s)

        async with (
            self._session_ctx() as session,
            session.post(self._base_url + path, **post_kwargs) as resp,
        ):
            if resp.status >= 400:
                # The stub returns JSON ``{"error": ..., "message": ...}``
                # on adapter failures; surface that in the raised
                # error so the smoke / coordinator log shows the
                # actual cause instead of an opaque 500.
                payload: dict[str, Any] = {}
                with suppress(Exception):
                    payload = await resp.json() or {}
                err = payload.get("error") or "stub_error"
                msg = payload.get("message") or resp.reason or ""
                raise StubResponseError(
                    f"stub {path} returned {resp.status}: {err}: {msg}",
                    status=resp.status, error=str(err), message_text=str(msg),
                )
            data = await resp.json()
            return data if isinstance(data, dict) else {"value": data}


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint parsing
# ──────────────────────────────────────────────────────────────────────────────


class StubEndpointInvalid(XRLEnvError):
    """Raised when a SandboxHandle's stub_endpoint can't be parsed."""


def _parse_endpoint(endpoint: str) -> tuple[str, str | None, str]:
    """Return ``(kind, uds_path, base_url)``.

    - ``unix:///path/to/sock`` → ``("uds", "/path/to/sock", "http://stub")``
    - ``tcp://host:port`` → ``("tcp", None, "http://host:port")``
    """
    parsed = urlparse(endpoint)
    if parsed.scheme == "unix":
        return ("uds", parsed.path, "http://stub")
    if parsed.scheme == "tcp":
        if not parsed.hostname or parsed.port is None:
            raise StubEndpointInvalid(f"tcp endpoint must include host:port; got {endpoint!r}")
        return ("tcp", None, f"http://{parsed.hostname}:{parsed.port}")
    raise StubEndpointInvalid(f"unsupported stub endpoint scheme: {endpoint!r}")
