"""Pre-flight checks that run BEFORE a rollout acquires anything or spends anything.

The check here exists because of a specific, expensive failure: an 89-task sweep ran to
completion scoring 0.000 on every task, because the agent in each container dialled a gateway
that wasn't there. Every trial installed cleanly, started work, exhausted its LLM retries, and
recorded a clean "completed" with zero tokens. ~25 minutes of cluster time and 89 containers to
learn one thing a single TCP connect answers in milliseconds.

The endpoint was wrong for a reason worth naming: ``.env`` held the right URL, but a **stale
export in the launching shell outranked it** — :func:`beagle.dotenv.load_project_dotenv` fills
gaps only, so host env wins. Re-running the script that rewrites ``.env`` didn't help, because it
can't touch an already-exported variable. ``.env`` looked correct every time it was inspected
while every run kept using the stale value. :func:`gateway_mismatch_hint` exists to say that out
loud rather than leave it to be deduced.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from beagle.agents.core.litellm_gateway import LOCAL_PROXY_ENV

#: Re-exported from the module that owns this deployment's gateway knowledge — the name is not
#: re-declared here. Only the ``internal`` route involves it; see :func:`gateway_target`.
INTERNAL_PROXY_ENV = LOCAL_PROXY_ENV
#: How long to wait for a TCP connect. Generous for a healthy local/VPC endpoint, short enough
#: that an unreachable one fails fast — this runs before every live evaluate.
DEFAULT_TIMEOUT_S = 5.0


def gateway_target(cfg: Any) -> tuple[str, str] | None:
    """``(url, source)`` for the run's LLM endpoint, or ``None`` when there is nothing to probe.

    ``source`` is human-readable provenance — the env var or config path the URL came from — so
    a failure can name *where to go fix it*, which is the part that took longest to work out
    when this actually happened.

    **Coverage is by ROUTE TYPE, not by vendor** — nothing here is specific to one org's
    gateway:

    * ``gateway`` — any OpenAI-compatible endpoint declared in config as ``api_base``. Fully
      generic, and the route an external user configures, so the check is real for them too.
    * ``internal`` — a deployment-native route; the URL comes from whatever
      :func:`~beagle.agents.core.litellm_gateway.gateway_litellm_kwargs` resolves.
    * ``direct`` — ``None``. A first-party vendor API is not an endpoint we own or operate, a
      TCP probe of it proves little, and it would false-fail wherever egress is proxied. This
      no-op is deliberate, not an oversight.
    """
    from beagle.agents.core.litellm_gateway import gateway_litellm_kwargs
    from beagle.agents.core.provider import GatewayProvider, InternalProvider, provider_config

    try:
        route = provider_config(dict(cfg.agent.config or {}))
    except (AttributeError, ValueError):
        return None
    if isinstance(route, GatewayProvider):
        base = route.extra_args.api_base
        return (base, "provider.extra_args.api_base in the config") if base else None
    if isinstance(route, InternalProvider):
        # Delegate to the shared resolver instead of re-reading an env var, so this tracks
        # whatever the deployment route actually resolves to.
        gw = gateway_litellm_kwargs()
        url = (gw or {}).get("api_base", "")
        return (url, f"${INTERNAL_PROXY_ENV} in the host environment") if url else None
    return None


def probe_tcp(url: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> str | None:
    """``None`` when a TCP connection to ``url``'s host:port succeeds; a short reason otherwise.

    Deliberately a bare TCP connect, not an HTTP request: it needs no credentials, spends no
    tokens, and cannot be confused by an auth or routing error. It answers exactly one question —
    *is anything listening where we are about to send every request?*

    **What it does not prove:** that a rollout CONTAINER can reach the endpoint. This runs on the
    launch host, so a container-to-host network partition would still slip through. It catches
    the far more common case — a wrong, stale, or dead address — which is what actually bit.
    """
    parts = urlsplit(url if "//" in url else f"//{url}")
    host, port = parts.hostname, parts.port
    if not host:
        return f"no host in {url!r}"
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return None
    except TimeoutError:
        return f"timed out after {timeout:g}s connecting to {host}:{port}"
    except OSError as exc:
        return f"{exc.strerror or exc} connecting to {host}:{port}"


def gateway_mismatch_hint(url: str, source: str) -> str:
    """The extra line to print when the unreachable URL came from the host env AND ``.env``
    disagrees — i.e. a stale export is silently beating a correct ``.env``.

    This is the single most useful thing to say in that situation, because every other signal
    points the wrong way: ``.env`` reads correct, and the script that maintains it reports
    success. ``""`` when there's nothing to add.
    """
    if INTERNAL_PROXY_ENV not in source:
        return ""
    from beagle.dotenv import find_dotenv, parse_dotenv

    path = find_dotenv()
    if path is None or not Path(path).is_file():
        return ""
    try:
        in_file = (parse_dotenv(Path(path).read_text(encoding="utf-8")).get(INTERNAL_PROXY_ENV)
                   or "").strip()
    except OSError:
        return ""
    if not in_file or in_file.rstrip("/") == (url or "").rstrip("/"):
        return ""
    return (
        f"\n  {path.name} says {in_file} — DIFFERENT, and it was ignored: a variable already "
        f"exported in your shell takes precedence over {path.name}, which fills gaps only. "
        f"Re-running the script that rewrites {path.name} will NOT fix this.\n"
        f"  Fix the shell:  unset {INTERNAL_PROXY_ENV}   (or export the correct value)"
    )


def check_gateway(cfg: Any, *, timeout: float = DEFAULT_TIMEOUT_S) -> tuple[str, str, str | None] | None:
    """``(url, source, failure)`` for the run's endpoint — ``failure`` is ``None`` when reachable.

    ``None`` when there is no endpoint to probe (a direct route). Callers decide what to do:
    the dry run reports, a live run refuses to start.
    """
    target = gateway_target(cfg)
    if target is None:
        return None
    url, source = target
    return url, source, probe_tcp(url, timeout=timeout)


def require_gateway_reachable(cfg: Any, *, timeout: float = DEFAULT_TIMEOUT_S) -> None:
    """Raise :class:`SystemExit` if the run's LLM endpoint is unreachable. No-op otherwise.

    Called on the live path before any container is acquired, so an unreachable gateway costs one
    failed connect instead of a full sweep of zero-scoring trials.
    """
    checked = check_gateway(cfg, timeout=timeout)
    if checked is None:
        return
    url, source, failure = checked
    if failure is None:
        return
    raise SystemExit(
        f"[beagle evaluate] PREFLIGHT FAILED: the LLM gateway is unreachable.\n"
        f"  endpoint: {url}\n"
        f"  error   : {failure}\n"
        f"  from    : {source}"
        f"{gateway_mismatch_hint(url, source)}\n"
        f"  Nothing was acquired and nothing was spent. Every trial would have scored 0 with an "
        f"agent that never reached a model.\n"
        f"  Re-check with: beagle evaluate --config <cfg> --dry-run"
    )


__all__ = ["INTERNAL_PROXY_ENV", "check_gateway", "gateway_mismatch_hint", "gateway_target",
           "probe_tcp", "require_gateway_reachable"]
