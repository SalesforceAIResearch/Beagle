"""Unit tests for the control-plane tag→digest resolver (freshness model).

Covers the resolution happy path, the fresh-TTL cache, and the
operator-chosen failure semantics: last-known-good within a bounded
stale window on a transient outage, else fail; permanent (4xx) failures
never serve a stale digest; explicit digests + non-registry refs pass
through.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from xrlenv.control.registry_resolver import (
    RegistryDigestResolver,
    RegistryResolveError,
    _parse_host_map,
    _PermanentResolveError,
    parse_registry_tag_ref,
    resolver_from_env,
)

DIGEST = "sha256:" + "a" * 64
DIGEST2 = "sha256:" + "b" * 64
REF = "ip-10-0-5-6:5011/wai/substrate:1ca77813"
PINNED = f"ip-10-0-5-6:5011/wai/substrate@{DIGEST}"


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _resolver(*, fetch, clock=None, fresh_ttl_s=60.0, max_stale_s=900.0):
    return RegistryDigestResolver(
        fresh_ttl_s=fresh_ttl_s,
        max_stale_s=max_stale_s,
        manifest_digest_fn=fetch,
        clock=clock or _Clock(),
    )


# ── parse_registry_tag_ref ──────────────────────────────────────────────────


def test_parse_registry_qualified_tag() -> None:
    assert parse_registry_tag_ref(REF) == (
        "ip-10-0-5-6:5011", "wai/substrate", "1ca77813",
    )


def test_parse_localhost_and_default_tag() -> None:
    assert parse_registry_tag_ref("localhost:5000/a/b") == (
        "localhost:5000", "a/b", "latest",
    )


def test_parse_returns_none_for_passthrough_refs() -> None:
    assert parse_registry_tag_ref(PINNED) is None  # already digest
    assert parse_registry_tag_ref("ubuntu:22.04") is None  # bare name
    assert parse_registry_tag_ref("library/python:3.12") is None  # docker-hub


# ── resolve() ───────────────────────────────────────────────────────────────


async def test_resolve_passes_through_digest_and_bare_refs() -> None:
    calls: list[tuple[str, str, str]] = []

    async def _fetch(host, repo, tag):
        calls.append((host, repo, tag))
        return DIGEST

    r = _resolver(fetch=_fetch)
    assert await r.resolve(PINNED) == PINNED
    assert await r.resolve("ubuntu:22.04") == "ubuntu:22.04"
    assert calls == []  # nothing to resolve → no registry probe


async def test_resolve_pins_tag_to_digest() -> None:
    async def _fetch(host, repo, tag):
        return DIGEST

    r = _resolver(fetch=_fetch)
    assert await r.resolve(REF) == PINNED


async def test_resolve_caches_within_fresh_ttl() -> None:
    n = {"calls": 0}

    async def _fetch(host, repo, tag):
        n["calls"] += 1
        return DIGEST

    clock = _Clock()
    r = _resolver(fetch=_fetch, clock=clock, fresh_ttl_s=60.0)
    assert await r.resolve(REF) == PINNED
    clock.advance(30.0)  # within TTL
    assert await r.resolve(REF) == PINNED
    assert n["calls"] == 1  # served from cache, no re-probe
    clock.advance(40.0)  # now past the 60s TTL
    assert await r.resolve(REF) == PINNED
    assert n["calls"] == 2  # re-probed


async def test_resolve_transient_failure_serves_last_known_good() -> None:
    """A registry blip after a successful resolution serves the LKG digest
    while within the stale window."""
    state = {"fail": False}

    async def _fetch(host, repo, tag):
        if state["fail"]:
            raise ConnectionError("registry down")
        return DIGEST

    clock = _Clock()
    r = _resolver(fetch=_fetch, clock=clock, fresh_ttl_s=60.0, max_stale_s=900.0)
    assert await r.resolve(REF) == PINNED  # warms the LKG cache
    state["fail"] = True
    clock.advance(120.0)  # past fresh TTL → re-probe attempted, fails
    assert await r.resolve(REF) == PINNED  # served LKG


async def test_resolve_transient_failure_past_stale_window_raises() -> None:
    state = {"fail": False}

    async def _fetch(host, repo, tag):
        if state["fail"]:
            raise ConnectionError("registry down")
        return DIGEST

    clock = _Clock()
    r = _resolver(fetch=_fetch, clock=clock, fresh_ttl_s=60.0, max_stale_s=300.0)
    assert await r.resolve(REF) == PINNED
    state["fail"] = True
    clock.advance(400.0)  # past the 300s stale window
    with pytest.raises(RegistryResolveError):
        await r.resolve(REF)


async def test_resolve_transient_failure_no_prior_lkg_raises() -> None:
    async def _fetch(host, repo, tag):
        raise ConnectionError("registry down")

    r = _resolver(fetch=_fetch)
    with pytest.raises(RegistryResolveError):
        await r.resolve(REF)


async def test_resolve_permanent_failure_never_serves_stale() -> None:
    """A 404 (bad/deleted tag) is not an outage — never serve the LKG."""
    state = {"perm": False}

    async def _fetch(host, repo, tag):
        if state["perm"]:
            raise _PermanentResolveError("404 tag gone")
        return DIGEST

    clock = _Clock()
    r = _resolver(fetch=_fetch, clock=clock, fresh_ttl_s=1.0, max_stale_s=900.0)
    assert await r.resolve(REF) == PINNED  # warm LKG
    state["perm"] = True
    clock.advance(10.0)  # past TTL → re-probe → permanent error
    with pytest.raises(_PermanentResolveError):
        await r.resolve(REF)


# ── resolver_from_env ───────────────────────────────────────────────────────


def test_resolver_from_env_disabled() -> None:
    assert resolver_from_env({"XRLENV_REGISTRY_DIGEST_RESOLVE": "0"}) is None
    assert resolver_from_env({"XRLENV_REGISTRY_DIGEST_RESOLVE": "off"}) is None


def test_resolver_from_env_defaults_on() -> None:
    r = resolver_from_env({})
    assert isinstance(r, RegistryDigestResolver)
    assert r._scheme == "http"
    assert r._fresh_ttl_s == 60.0
    assert r._max_stale_s == 900.0


def test_resolver_from_env_overrides() -> None:
    r = resolver_from_env({
        "XRLENV_REGISTRY_SCHEME": "https",
        "XRLENV_REGISTRY_RESOLVE_TTL_S": "10",
        "XRLENV_REGISTRY_RESOLVE_MAX_STALE_S": "120",
    })
    assert isinstance(r, RegistryDigestResolver)
    assert r._scheme == "https"
    assert r._fresh_ttl_s == 10.0
    assert r._max_stale_s == 120.0


# ── dial-host map (co-located CP probes the registry over loopback) ──────────


def test_parse_host_map() -> None:
    assert _parse_host_map("") == {}
    assert _parse_host_map("ip-10-0-5-6:5011=127.0.0.1:5011") == {
        "ip-10-0-5-6:5011": "127.0.0.1:5011",
    }
    # multiple entries + surrounding whitespace tolerated
    assert _parse_host_map(" a:1=b:2 , c:3=d:4 ") == {"a:1": "b:2", "c:3": "d:4"}
    # malformed entries (no '=') are skipped, not fatal
    assert _parse_host_map("garbage,no-equals,x:1=y:2") == {"x:1": "y:2"}


def test_resolver_from_env_host_map() -> None:
    r = resolver_from_env({
        "XRLENV_REGISTRY_RESOLVE_HOST_MAP":
            "ip-10-0-5-6:5011=127.0.0.1:5011,ip-10-0-5-6:5010=127.0.0.1:5010",
    })
    assert isinstance(r, RegistryDigestResolver)
    assert r._resolve_host_map == {
        "ip-10-0-5-6:5011": "127.0.0.1:5011",
        "ip-10-0-5-6:5010": "127.0.0.1:5010",
    }


def test_resolver_from_env_no_host_map_default() -> None:
    r = resolver_from_env({})
    assert isinstance(r, RegistryDigestResolver)
    assert r._resolve_host_map == {}


def _install_recording_httpx(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Patch ``httpx.AsyncClient`` with a stub that records the probed URL and
    returns a fixed ``Docker-Content-Digest``. Returns the dict the test reads
    the dialed URL back from (``probed["url"]``)."""
    import httpx

    probed: dict[str, str] = {}

    class _Resp:
        status_code = 200
        is_success = True
        headers: ClassVar[dict[str, str]] = {"Docker-Content-Digest": DIGEST}

        def raise_for_status(self) -> None:
            pass

    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *a: object) -> bool:
            return False

        async def head(self, url: str, headers: object = None) -> _Resp:
            probed["url"] = url
            return _Resp()

        async def get(self, url: str, headers: object = None) -> _Resp:
            probed["url"] = url
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return probed


async def test_dial_host_map_probes_loopback_but_pins_external_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control plane probes the loopback address while the returned digest
    ref keeps the externally-routable host the remote nodes pull from."""
    probed = _install_recording_httpx(monkeypatch)
    r = RegistryDigestResolver(
        resolve_host_map={"ip-10-0-5-6:5011": "127.0.0.1:5011"},
    )
    out = await r.resolve(REF)
    # probe dialed loopback…
    assert probed["url"] == (
        "http://127.0.0.1:5011/v2/wai/substrate/manifests/1ca77813"
    )
    # …but the recorded/returned ref keeps the external host for the nodes.
    assert out == PINNED


async def test_no_host_map_dials_ref_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no map entry, the probe dials the ref host verbatim (default)."""
    probed = _install_recording_httpx(monkeypatch)
    r = RegistryDigestResolver()  # empty map
    out = await r.resolve(REF)
    assert probed["url"] == (
        "http://ip-10-0-5-6:5011/v2/wai/substrate/manifests/1ca77813"
    )
    assert out == PINNED


# ── public-registry probe: http→https redirect + anonymous bearer token ───────
#
# public.ecr.aws (DeepSWE's prebuilt images) 301s an http probe to https and then
# 401-challenges it with a Bearer realm; the resolver must follow the redirect and
# do the anonymous-token dance (the node's ``docker pull`` does the same natively).

PUB_REF = "public.ecr.aws/d3j8x8q7/swe-bench-202605:kh-v1.1"
PUB_PINNED = f"public.ecr.aws/d3j8x8q7/swe-bench-202605@{DIGEST}"
_CHALLENGE = (
    'Bearer realm="https://public.ecr.aws/token/",'
    'service="public.ecr.aws",scope="aws"'
)


def _install_mock_httpx(monkeypatch: pytest.MonkeyPatch, handler: object) -> None:
    """Inject an ``httpx.MockTransport(handler)`` into the *real* AsyncClient the
    resolver constructs (so ``follow_redirects`` + head/get + the token GET all run
    through the mock, deterministically, with no network)."""
    import httpx

    real = httpx.AsyncClient

    def _factory(*a: object, **k: object) -> httpx.AsyncClient:
        k.setdefault("transport", httpx.MockTransport(handler))  # type: ignore[arg-type]
        return real(*a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", _factory)


def _ecr_handler(seen: list[str]):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        u = request.url
        if u.scheme == "http":  # http → https redirect (301)
            return httpx.Response(
                301, headers={"Location": str(u.copy_with(scheme="https"))},
            )
        if "/token" in u.path:  # anonymous token endpoint
            assert u.params.get("service") == "public.ecr.aws"
            assert u.params.get("scope") == "aws"  # challenge scope honored verbatim
            return httpx.Response(200, json={"token": "TOK"})
        # manifest endpoint (https): challenge until the bearer token is presented
        if request.headers.get("Authorization") == "Bearer TOK":
            return httpx.Response(200, headers={"Docker-Content-Digest": DIGEST})
        return httpx.Response(401, headers={"WWW-Authenticate": _CHALLENGE})

    return handler


async def test_public_registry_follows_redirect_and_bearer_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[str] = []
    _install_mock_httpx(monkeypatch, _ecr_handler(recorded))

    r = RegistryDigestResolver()  # scheme=http default → exercises the redirect
    out = await r.resolve(PUB_REF)
    assert out == PUB_PINNED
    # the flow hit: http manifest (301) → https manifest (401) → token → https (200)
    assert any(x.startswith("http://public.ecr.aws/v2/") for x in recorded)
    assert any("/token" in x for x in recorded)


async def test_private_http_registry_needs_no_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain private HTTP registry answers 200 directly — no redirect, no 401 —
    so the token endpoint is never hit (no regression / no extra round-trip)."""
    import httpx

    recorded: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(str(request.url))
        return httpx.Response(200, headers={"Docker-Content-Digest": DIGEST})

    _install_mock_httpx(monkeypatch, handler)
    r = RegistryDigestResolver()
    out = await r.resolve(REF)
    assert out == PINNED
    assert recorded == ["http://ip-10-0-5-6:5011/v2/wai/substrate/manifests/1ca77813"]
    assert not any("/token" in x for x in recorded)


async def test_public_registry_404_is_permanent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.scheme == "http":
            return httpx.Response(
                301, headers={"Location": str(request.url.copy_with(scheme="https"))},
            )
        return httpx.Response(404)  # tag genuinely gone

    _install_mock_httpx(monkeypatch, handler)
    r = RegistryDigestResolver()
    with pytest.raises(_PermanentResolveError):
        await r.resolve(PUB_REF)
