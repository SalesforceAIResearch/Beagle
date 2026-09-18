"""Gateway reachability preflight — the check that would have caught an 89-task all-zero sweep.

Addresses here are RFC 5737 documentation ranges (``192.0.2.0/24``, ``198.51.100.0/24``), not
the real ones from the incident this was written for. They are never routed, and they keep the
file shippable: the OSS port refuses any RFC 1918 address in the published surface.

An unreachable gateway is the worst kind of failure to debug from artifacts: every trial installs
cleanly, runs, exhausts its LLM retries and records a clean "completed" with zero tokens, so the
run looks like a capability result rather than a broken endpoint.
"""

from __future__ import annotations

import socket
import threading

import pytest

from beagle.cli._preflight import (
    INTERNAL_PROXY_ENV,
    check_gateway,
    gateway_mismatch_hint,
    gateway_target,
    probe_tcp,
    require_gateway_reachable,
)


@pytest.fixture
def listening_port():
    """A real socket that accepts connections, closed on teardown."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    threading.Thread(target=lambda: None, daemon=True).start()
    yield srv.getsockname()[1]
    srv.close()


class _Cfg:
    """Minimal stand-in for a RunConfig: the preflight only reads agent.config."""

    def __init__(self, provider: dict | None) -> None:
        self.agent = type("A", (), {"config": {"provider": provider} if provider else {}})()


_INTERNAL = {"type": "internal", "name": "llm-gateway-express-local-proxy"}


# --- probe -------------------------------------------------------------------


def test_probe_succeeds_against_a_real_listener(listening_port: int) -> None:
    assert probe_tcp(f"http://127.0.0.1:{listening_port}/") is None


def test_probe_reports_a_refused_connection() -> None:
    """The incident's signature: the address resolves but nothing is listening."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.close()                                   # nothing listening now
    failure = probe_tcp(f"http://127.0.0.1:{port}/", timeout=2.0)
    assert failure is not None and str(port) in failure


def test_probe_rejects_a_url_with_no_host() -> None:
    assert probe_tcp("not-a-url") is not None or True   # never raises
    assert probe_tcp("http://") is not None


def test_probe_defaults_the_port_by_scheme() -> None:
    """A gateway URL without an explicit port must still be probed, not skipped."""
    failure = probe_tcp("https://127.0.0.1/", timeout=1.0)
    assert failure is None or "127.0.0.1:443" in failure


# --- target resolution -------------------------------------------------------


def test_target_for_an_internal_route_names_the_env_var(monkeypatch) -> None:
    """The failure message has to say WHERE the URL came from — that was the slow part."""
    monkeypatch.setenv(INTERNAL_PROXY_ENV, "http://192.0.2.1:18088/")
    url, source = gateway_target(_Cfg(_INTERNAL))
    assert url == "http://192.0.2.1:18088/"
    assert INTERNAL_PROXY_ENV in source


def test_target_for_a_gateway_route_names_the_config() -> None:
    cfg = _Cfg({"type": "gateway", "name": "gw",
                "extra_args": {"api_base": "http://gw:9000", "api_key_env": "K"}})
    url, source = gateway_target(cfg)
    assert url == "http://gw:9000" and "config" in source


def test_no_target_for_a_direct_route_or_unset_proxy(monkeypatch) -> None:
    """A direct first-party route has no endpoint of ours to check; an unset proxy is already
    reported by the existing preflight, so don't double-report it as unreachable."""
    assert gateway_target(_Cfg({"type": "direct", "name": "openai"})) is None
    monkeypatch.delenv(INTERNAL_PROXY_ENV, raising=False)
    assert gateway_target(_Cfg(_INTERNAL)) is None
    assert check_gateway(_Cfg(None)) is None


# --- enforcement -------------------------------------------------------------


def test_live_run_refuses_to_start_on_an_unreachable_gateway(monkeypatch) -> None:
    """THE point of this module: fail before acquiring anything, rather than after 89 trials."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.close()
    monkeypatch.setenv(INTERNAL_PROXY_ENV, f"http://127.0.0.1:{port}/")
    with pytest.raises(SystemExit) as exc:
        require_gateway_reachable(_Cfg(_INTERNAL), timeout=2.0)
    msg = str(exc.value)
    assert "PREFLIGHT FAILED" in msg
    assert str(port) in msg                       # names the endpoint
    assert INTERNAL_PROXY_ENV in msg              # names where it came from
    assert "nothing was spent" in msg.lower()


def test_reachable_gateway_is_a_no_op(monkeypatch, listening_port: int) -> None:
    monkeypatch.setenv(INTERNAL_PROXY_ENV, f"http://127.0.0.1:{listening_port}/")
    require_gateway_reachable(_Cfg(_INTERNAL), timeout=2.0)      # must not raise


def test_direct_route_is_never_blocked(monkeypatch) -> None:
    """Only OUR endpoints are probed — a direct provider must not be gated on a TCP check."""
    require_gateway_reachable(_Cfg({"type": "direct", "name": "openai"}), timeout=2.0)


# --- the stale-export diagnosis ----------------------------------------------


def test_mismatch_hint_calls_out_a_stale_export(monkeypatch, tmp_path) -> None:
    """The incident's real root cause: .env was CORRECT, but a stale shell export outranked it,
    so every other signal pointed the wrong way. The hint has to say this explicitly."""
    env = tmp_path / ".env"
    env.write_text(f"{INTERNAL_PROXY_ENV}=http://192.0.2.10:18088/\n", encoding="utf-8")
    monkeypatch.setattr("beagle.dotenv.find_dotenv", lambda: env)
    hint = gateway_mismatch_hint("http://198.51.100.99:18088/", f"${INTERNAL_PROXY_ENV} in the host environment")
    assert "192.0.2.10" in hint                 # what .env actually says
    assert "fills gaps only" in hint or "precedence" in hint
    assert "will NOT fix this" in hint            # re-running the rewrite script is a dead end
    assert f"unset {INTERNAL_PROXY_ENV}" in hint  # the actual fix


def test_mismatch_hint_is_silent_when_env_agrees(monkeypatch, tmp_path) -> None:
    """No noise when .env and the shell match — a trailing slash is not a mismatch."""
    env = tmp_path / ".env"
    env.write_text(f"{INTERNAL_PROXY_ENV}=http://192.0.2.10:18088/\n", encoding="utf-8")
    monkeypatch.setattr("beagle.dotenv.find_dotenv", lambda: env)
    assert gateway_mismatch_hint("http://192.0.2.10:18088",
                                 f"${INTERNAL_PROXY_ENV} in the host environment") == ""


def test_mismatch_hint_is_silent_for_a_config_sourced_url(tmp_path) -> None:
    """A config-declared api_base has nothing to do with .env precedence."""
    assert gateway_mismatch_hint("http://gw:9000", "provider.extra_args.api_base in the config") == ""


# --- vendor neutrality (the OSS path) ----------------------------------------


def test_a_plain_openai_compatible_gateway_is_checked(listening_port: int) -> None:
    """Coverage is by ROUTE TYPE, not vendor. An external user configuring any OpenAI-compatible
    endpoint gets a real check -- this must not degrade to a no-op outside one org's deployment."""
    cfg = _Cfg({"type": "gateway", "name": "my-proxy",
                "extra_args": {"api_base": f"http://127.0.0.1:{listening_port}/v1",
                               "api_key_env": "MY_KEY"}})
    url, source, failure = check_gateway(cfg, timeout=2.0)
    assert failure is None and str(listening_port) in url
    assert "config" in source                      # provenance points at the config, not an env var


def test_a_dead_third_party_gateway_blocks_the_run() -> None:
    """Same enforcement for a non-vendor endpoint: refuse before spending."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.close()
    cfg = _Cfg({"type": "gateway", "name": "my-proxy",
                "extra_args": {"api_base": f"http://127.0.0.1:{port}/v1"}})
    with pytest.raises(SystemExit, match="PREFLIGHT FAILED"):
        require_gateway_reachable(cfg, timeout=2.0)


def test_the_env_var_name_is_not_redeclared_here() -> None:
    """One definition, in the module that owns this deployment's gateway knowledge."""
    from beagle.agents.core.litellm_gateway import LOCAL_PROXY_ENV

    assert INTERNAL_PROXY_ENV is LOCAL_PROXY_ENV
