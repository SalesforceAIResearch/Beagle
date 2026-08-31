"""Control-plane registry tag→digest resolver (freshness model, Part 2).

Turns a registry-qualified *tag* ref (``host:5011/repo:tag``) into a
content-addressed *digest* ref (``host:5011/repo@sha256:…``) by probing
the registry's manifest, so the node materializes exactly the bytes the
registry currently serves under that tag — even when the tag was
re-pushed under the same name (the mutable-tag staleness problem).

Consumers keep a stable, human-readable tag in their config forever
(``…/substrate:stable``); the control plane resolves + pins it per
acquire. The digest is recorded on the raw session, so a moving tag
stays auditable ("what actually ran").

**Failure semantics (operator-chosen).** On a *transient* registry
failure (connection refused / timeout — an outage, not a bad ref) the
resolver serves the **last-known-good** digest for that ref if one was
resolved within ``max_stale_s``; otherwise it raises
:class:`RegistryResolveError` and the acquire fails. This preserves
digest pinning + auditability across a registry blip without stalling a
whole training run, while never silently running an unverifiable image
once the stale window lapses. A *permanent* failure (the tag 404s — it
doesn't exist) fails immediately, never serving a stale digest.

An explicit ``repo@sha256:…`` ref, and any ref that isn't a
registry-qualified tag (bare names, Docker-Hub-relative repos), pass
through unchanged — there is nothing to resolve.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

LOGGER = logging.getLogger("xrlenv.control.registry_resolver")

# (host, repo, tag) -> "sha256:…"
ManifestDigestFn = Callable[[str, str, str], Awaitable[str]]

_MANIFEST_ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def resolver_from_env(
    env: dict[str, str] | None = None,
) -> RegistryDigestResolver | None:
    """Build a resolver from ``XRLENV_REGISTRY_*`` env, or ``None`` when
    disabled. Wired into the raw-container path by both runtimes.

    Knobs (all optional):

    - ``XRLENV_REGISTRY_DIGEST_RESOLVE`` — ``0``/``false``/``off`` disables
      the freshness model entirely (acquires use the ref verbatim, the
      legacy mutable-tag behavior). Default on. The kill-switch for a
      cluster whose control plane can't reach the registry, or that
      relies on a non-HTTP registry not configured here.
    - ``XRLENV_REGISTRY_SCHEME`` — ``http`` (default; the private
      insecure registry) or ``https``.
    - ``XRLENV_REGISTRY_RESOLVE_TTL_S`` — reuse a cached resolution
      without re-probing for this long (default 60s; collapses an
      acquire burst into one probe).
    - ``XRLENV_REGISTRY_RESOLVE_MAX_STALE_S`` — on a transient registry
      outage, serve the last-known-good digest up to this age before
      failing the acquire (default 900s).
    - ``XRLENV_REGISTRY_RESOLVE_HOST_MAP`` — comma-separated
      ``ref-host:port=dial-host:port`` pairs. Rewrites *only the host:port
      the control plane dials* when probing the manifest; the recorded /
      returned digest ref keeps the original ref host (what remote nodes
      pull from). The case this exists for: a control plane **co-located
      with the registry**. There, the box can reach the registry over
      loopback (``127.0.0.1:5011``) but often *not* via its own external
      name (``<registry-host>:5011``) — host→own-published-port hairpin
      NAT is unreliable (notably under docker 29), while remote nodes
      reach the external name fine. Set e.g.
      ``<registry-host>:5011=127.0.0.1:5011`` so the CP probes loopback
      while nodes still pull ``<registry-host>:5011/...@sha256:…``. Empty /
      unset → dial the ref host verbatim (the default).
    """
    import os

    src = env if env is not None else dict(os.environ)
    if src.get("XRLENV_REGISTRY_DIGEST_RESOLVE", "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        return None

    def _f(key: str, default: float) -> float:
        try:
            return float(src.get(key, default))
        except (TypeError, ValueError):
            return default

    return RegistryDigestResolver(
        scheme=src.get("XRLENV_REGISTRY_SCHEME", "http"),
        fresh_ttl_s=_f("XRLENV_REGISTRY_RESOLVE_TTL_S", 60.0),
        max_stale_s=_f("XRLENV_REGISTRY_RESOLVE_MAX_STALE_S", 900.0),
        resolve_host_map=_parse_host_map(
            src.get("XRLENV_REGISTRY_RESOLVE_HOST_MAP", ""),
        ),
    )


def _parse_host_map(spec: str) -> dict[str, str]:
    """Parse ``a=b,c=d`` into ``{"a": "b", "c": "d"}`` for the dial-host map.

    Maps the registry ``host:port`` embedded in an image ref (the address
    remote nodes pull from) to the ``host:port`` the *control plane* should
    dial when probing the manifest. Registry hosts carry a port colon but
    never ``=``, so splitting on the first ``=`` is unambiguous (IPv6 dial
    targets work too). Malformed / empty entries are skipped rather than
    raising — a typo'd map shouldn't take down acquires; it just falls back
    to dialing the ref host verbatim.
    """
    out: dict[str, str] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        ref_host, _, dial_host = entry.partition("=")
        ref_host, dial_host = ref_host.strip(), dial_host.strip()
        if ref_host and dial_host:
            out[ref_host] = dial_host
    return out


class RegistryResolveError(RuntimeError):
    """Raised to the acquire path when a ref cannot be pinned to a digest.

    Either the registry is unreachable and no last-known-good digest
    exists within the stale window, or the ref is permanently
    unresolvable (the tag 404s).
    """


class _PermanentResolveError(RegistryResolveError):
    """A 4xx from the registry — the ref is wrong / the tag is gone. Never
    eligible for the last-known-good fallback (it's not an outage)."""


def parse_registry_tag_ref(image_ref: str) -> tuple[str, str, str] | None:
    """Split a registry-qualified *tag* ref into ``(host, repo, tag)``.

    Returns ``None`` — meaning "nothing to resolve, pass through" — when
    the ref is already digest-pinned (``@sha256:…``), or has no registry
    host (a bare ``name:tag`` or a Docker-Hub-relative ``library/x:tag``
    we don't probe). Docker's rule for "is the first segment a registry
    host": it contains a ``.`` / ``:`` or equals ``localhost``.
    """
    if "@" in image_ref:
        return None  # already digest-pinned
    head, slash, rest = image_ref.partition("/")
    if not slash:
        return None  # bare name[:tag], no registry path
    if not ("." in head or ":" in head or head == "localhost"):
        return None  # Docker-Hub-relative repo, not a private registry
    host = head
    # The tag's colon lives in the LAST path segment; a host:port colon
    # is already in ``head``, never in ``rest``.
    last_slash = rest.rfind("/")
    last_seg = rest[last_slash + 1:]
    if ":" in last_seg:
        repo, _, tag = rest.rpartition(":")
    else:
        repo, tag = rest, "latest"
    if not repo or not tag:
        return None
    return host, repo, tag


@dataclass(frozen=True)
class _CacheEntry:
    digest_ref: str  # full "host/repo@sha256:…"
    resolved_at: float  # clock() at resolution


class RegistryDigestResolver:
    """Resolve registry-qualified tag refs to digest refs, with a fresh-TTL
    cache and a last-known-good-within-stale-window fallback.

    The resolver is an *optional* dependency of the raw-container path:
    when unwired, acquires pass their ref through unchanged (the legacy
    mutable-tag behavior). Per-ref single-flight locking collapses a
    burst of acquires for the same ref into one registry probe.
    """

    def __init__(
        self,
        *,
        scheme: str = "http",
        fresh_ttl_s: float = 60.0,
        max_stale_s: float = 900.0,
        request_timeout_s: float = 10.0,
        manifest_digest_fn: ManifestDigestFn | None = None,
        clock: Callable[[], float] = time.monotonic,
        resolve_host_map: dict[str, str] | None = None,
    ) -> None:
        self._scheme = scheme
        self._fresh_ttl_s = fresh_ttl_s
        self._max_stale_s = max_stale_s
        self._request_timeout_s = request_timeout_s
        self._fetch: ManifestDigestFn = (
            manifest_digest_fn or self._http_manifest_digest
        )
        self._clock = clock
        # ref host:port -> dial host:port for the manifest probe only. The
        # returned digest_ref always keeps the ref host (see ``resolve``), so
        # remote nodes pull the externally-routable address regardless of what
        # the control plane dialed locally.
        self._resolve_host_map = resolve_host_map or {}
        self._cache: dict[str, _CacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, ref: str) -> asyncio.Lock:
        lock = self._locks.get(ref)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[ref] = lock
        return lock

    async def resolve(self, image_ref: str) -> str:
        """Return ``image_ref`` pinned to a digest, or unchanged when there
        is nothing to resolve. Raises :class:`RegistryResolveError` when a
        registry-qualified tag can't be pinned."""
        parsed = parse_registry_tag_ref(image_ref)
        if parsed is None:
            return image_ref
        host, repo, tag = parsed

        async with self._lock_for(image_ref):
            now = self._clock()
            cached = self._cache.get(image_ref)
            if cached is not None and (now - cached.resolved_at) < self._fresh_ttl_s:
                return cached.digest_ref

            try:
                digest = await self._fetch(host, repo, tag)
            except _PermanentResolveError:
                # A bad / deleted tag is not an outage — never serve a
                # stale digest for it. Surface immediately.
                raise
            except Exception as exc:
                # Transient: registry unreachable. Serve last-known-good if
                # it's within the bounded stale window; otherwise fail.
                if (
                    cached is not None
                    and (now - cached.resolved_at) < self._max_stale_s
                ):
                    LOGGER.warning(
                        "registry resolve failed for %s (%s: %s); serving "
                        "last-known-good digest from %.0fs ago (stale window "
                        "%.0fs)",
                        image_ref, type(exc).__name__, exc,
                        now - cached.resolved_at, self._max_stale_s,
                    )
                    return cached.digest_ref
                raise RegistryResolveError(
                    f"cannot resolve {image_ref!r} to a digest: registry "
                    f"unreachable ({type(exc).__name__}: {exc}) and no "
                    f"last-known-good digest within {self._max_stale_s:.0f}s"
                ) from exc

            digest_ref = f"{host}/{repo}@{digest}"
            self._cache[image_ref] = _CacheEntry(
                digest_ref=digest_ref, resolved_at=now,
            )
            return digest_ref

    async def _http_manifest_digest(
        self, host: str, repo: str, tag: str,
    ) -> str:
        """Probe ``{scheme}://{host}/v2/{repo}/manifests/{tag}`` for the
        ``Docker-Content-Digest`` header. HEAD first; fall back to GET for
        registries that don't surface the header on HEAD.

        Follows HTTP→HTTPS redirects (``follow_redirects=True``) and does the
        standard **anonymous bearer-token** dance for a *public* registry that
        challenges an unauthenticated manifest probe — ``public.ecr.aws``
        (301 to https, then a ``Bearer`` challenge), ``ghcr.io``, ``quay.io``,
        etc. Without this, a public-ECR ref (e.g. DeepSWE's prebuilt images)
        fails the acquire even though the node can ``docker pull`` it fine (the
        docker daemon does the same redirect + token dance natively).

        The dialed host:port may differ from the ref ``host`` when an entry
        in ``resolve_host_map`` redirects it (e.g. a co-located control plane
        probing the registry over loopback). The digest this returns is
        content-addressed, so it's identical whichever address served it."""
        import httpx

        dial_host = self._resolve_host_map.get(host, host)
        url = f"{self._scheme}://{dial_host}/v2/{repo}/manifests/{tag}"
        accept = {"Accept": _MANIFEST_ACCEPT}
        auth: dict[str, str] = {}
        async with httpx.AsyncClient(
            timeout=self._request_timeout_s, follow_redirects=True,
        ) as client:

            async def _once(method: str, hdrs: dict[str, str]) -> httpx.Response:
                if method == "HEAD":
                    return await client.head(url, headers=hdrs)
                return await client.get(url, headers=hdrs)

            async def _request(method: str) -> httpx.Response:
                # One anonymous-bearer retry: a public registry answers an
                # unauthenticated probe with 401 + a ``WWW-Authenticate: Bearer
                # realm=…,service=…,scope=…`` challenge; fetch an anonymous token
                # from the realm and retry once. A private HTTP registry never
                # challenges, so this is a no-op there.
                resp = await _once(method, {**accept, **auth})
                if resp.status_code == 401 and not auth:
                    token = await self._anonymous_bearer_token(
                        client, resp.headers.get("WWW-Authenticate", ""), repo,
                    )
                    if token:
                        auth["Authorization"] = f"Bearer {token}"
                        resp = await _once(method, {**accept, **auth})
                return resp

            r = await _request("HEAD")
            have_digest = "docker-content-digest" in {
                k.lower() for k in r.headers
            }
            if r.status_code in (403, 405) or (r.is_success and not have_digest):
                # Registry refuses HEAD or omits the digest header on it.
                r = await _request("GET")
            if 400 <= r.status_code < 500:
                raise _PermanentResolveError(
                    f"registry {host} returned {r.status_code} for "
                    f"{repo}:{tag} (tag missing or unauthorized)"
                )
            r.raise_for_status()
            digest = r.headers.get("Docker-Content-Digest")
            if not digest:
                raise RegistryResolveError(
                    f"registry {host} returned no Docker-Content-Digest "
                    f"for {repo}:{tag}"
                )
            return str(digest)

    @staticmethod
    async def _anonymous_bearer_token(
        client: httpx.AsyncClient, www_authenticate: str, repo: str,
    ) -> str | None:
        """Fetch an anonymous pull token for a ``Bearer``-challenged registry.

        Parses ``Bearer realm="…",service="…",scope="…"`` and GETs the realm
        with the challenge's ``service`` + ``scope`` (public.ecr.aws uses
        ``scope="aws"``; most others ``repository:<repo>:pull`` — we honor the
        challenge's scope verbatim, defaulting to ``repository:<repo>:pull``
        when absent). Returns the token, or ``None`` when the challenge isn't a
        Bearer one / carries no realm (an anonymous retry then isn't possible)."""
        if not www_authenticate.strip().lower().startswith("bearer"):
            return None
        params = dict(re.findall(r'(\w+)="([^"]*)"', www_authenticate))
        realm = params.get("realm")
        if not realm:
            return None
        query: dict[str, str] = {}
        if params.get("service"):
            query["service"] = params["service"]
        query["scope"] = params.get("scope") or f"repository:{repo}:pull"
        tr = await client.get(realm, params=query)
        tr.raise_for_status()
        data = tr.json()
        token = data.get("token") or data.get("access_token")
        return str(token) if token else None
