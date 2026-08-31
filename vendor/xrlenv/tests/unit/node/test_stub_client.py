"""Tests for xrlenv/node/stub_client.py — _parse_endpoint + HTTP helpers."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from xrlenv.node.stub_client import StubClient, StubEndpointInvalid, _parse_endpoint

# ── _parse_endpoint ───────────────────────────────────────────────────────────

def test_parse_endpoint_unix() -> None:
    kind, path, base = _parse_endpoint("unix:///var/run/stub.sock")
    assert kind == "uds"
    assert path == "/var/run/stub.sock"
    assert base == "http://stub"


def test_parse_endpoint_tcp() -> None:
    kind, path, base = _parse_endpoint("tcp://127.0.0.1:9000")
    assert kind == "tcp"
    assert path is None
    assert base == "http://127.0.0.1:9000"


def test_parse_endpoint_tcp_missing_port_raises() -> None:
    with pytest.raises(StubEndpointInvalid, match="host:port"):
        _parse_endpoint("tcp://localhost")


def test_parse_endpoint_unknown_scheme_raises() -> None:
    with pytest.raises(StubEndpointInvalid, match="unsupported"):
        _parse_endpoint("grpc://localhost:50051")


def test_parse_endpoint_empty_string_raises() -> None:
    with pytest.raises(StubEndpointInvalid):
        _parse_endpoint("not-a-url")


# ── StubClient against a live in-process aiohttp server ──────────────────────


def _build_mini_app() -> web.Application:
    app = web.Application()

    async def healthz(_req: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def echo_post(req: web.Request) -> web.Response:
        body = await req.json()
        return web.json_response({"echo": body})

    app.router.add_get("/healthz", healthz)
    app.router.add_post("/echo", echo_post)
    return app


async def test_stub_client_healthz() -> None:
    async with TestClient(TestServer(_build_mini_app())) as tc:
        port = tc.port
        endpoint = f"tcp://127.0.0.1:{port}"
        client = StubClient(endpoint)
        result = await client.healthz()
        assert result["status"] == "ok"
        await client.close()


async def test_stub_client_post_json() -> None:
    async with TestClient(TestServer(_build_mini_app())) as tc:
        port = tc.port
        endpoint = f"tcp://127.0.0.1:{port}"
        async with StubClient(endpoint) as client:
            result = await client._post_json("/echo", {"key": "value"})
        assert result["echo"] == {"key": "value"}


async def test_stub_client_reuses_session_across_calls() -> None:
    async with TestClient(TestServer(_build_mini_app())) as tc:
        port = tc.port
        endpoint = f"tcp://127.0.0.1:{port}"
        async with StubClient(endpoint) as client:
            r1 = await client.healthz()
            r2 = await client.healthz()
        assert r1["status"] == r2["status"] == "ok"


# ── D17 stage 2 — per-call request_timeout_s overrides session default ───────


def _build_slow_app(sleep_s: float) -> web.Application:
    """Mini-app where every env_* endpoint sleeps before responding.
    Used to exercise the per-call timeout override: a tighter
    ``request_timeout_s`` than ``sleep_s`` must trip the per-call
    ``aiohttp.ClientTimeout`` (TimeoutError), and a generous one
    must let the call complete normally.
    """
    app = web.Application()

    async def slow(_req: web.Request) -> web.Response:
        await asyncio.sleep(sleep_s)
        return web.json_response({"obs": "done"})

    app.router.add_post("/env/setup", slow)
    app.router.add_post("/env/step", slow)
    app.router.add_post("/env/teardown", slow)
    return app


async def test_env_step_per_call_timeout_overrides_session_default() -> None:
    """A5 / D17 stage 2 acceptance: a per-call ``request_timeout_s``
    set on a single :py:meth:`StubClient.env_step` call applies to
    that aiohttp request only, even though the StubClient was
    constructed with a much wider session-level cap. This is the
    behavior the spec acceptance test asks for ("0.1 s ``step_timeout_s``
    surfaces the per-call ``aiohttp.ClientTimeout`` instead of the
    per-sandbox cap").
    """
    async with TestClient(TestServer(_build_slow_app(sleep_s=1.0))) as tc:
        port = tc.port
        endpoint = f"tcp://127.0.0.1:{port}"
        # Session default is generous (10 s) — without the per-call
        # override the call would complete normally; with it, the
        # 0.1 s cap fires first.
        async with StubClient(endpoint, request_timeout_s=10.0) as client:
            with pytest.raises(asyncio.TimeoutError):
                await client.env_step({"cmd": "noop"}, request_timeout_s=0.1)


async def test_env_setup_per_call_timeout_overrides_session_default() -> None:
    async with TestClient(TestServer(_build_slow_app(sleep_s=1.0))) as tc:
        port = tc.port
        endpoint = f"tcp://127.0.0.1:{port}"
        async with StubClient(endpoint, request_timeout_s=10.0) as client:
            with pytest.raises(asyncio.TimeoutError):
                await client.env_setup(
                    adapter_module="m", adapter_class="C", init_params={},
                    sandbox_id="sb", request_timeout_s=0.1,
                )


async def test_env_teardown_per_call_timeout_overrides_session_default() -> None:
    async with TestClient(TestServer(_build_slow_app(sleep_s=1.0))) as tc:
        port = tc.port
        endpoint = f"tcp://127.0.0.1:{port}"
        async with StubClient(endpoint, request_timeout_s=10.0) as client:
            with pytest.raises(asyncio.TimeoutError):
                await client.env_teardown(request_timeout_s=0.1)


async def test_env_step_without_per_call_timeout_uses_session_default() -> None:
    """No per-call override → session default applies (the stage-1
    per-sandbox cap, in production). Sleep < session cap completes
    normally; sleep > session cap would TimeoutError. Pinning the
    "no override" path so a regression that always overrides
    silently is caught.
    """
    async with TestClient(TestServer(_build_slow_app(sleep_s=0.05))) as tc:
        port = tc.port
        endpoint = f"tcp://127.0.0.1:{port}"
        async with StubClient(endpoint, request_timeout_s=2.0) as client:
            result = await client.env_step({"cmd": "noop"})
            assert result == {"obs": "done"}
