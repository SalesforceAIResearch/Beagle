"""Reach an OpenAI-compatible LLM gateway from a **LiteLLM-backed** agent — shared infrastructure.

LiteLLM is the model layer many agents drive (mini-swe today, more later), so *how* to point it at
a gateway lives HERE, once — not re-implemented per agent. A gateway is any **OpenAI-compatible
proxy**: one ``api_base`` serves any model — gpt *or* claude — over the OpenAI wire shape, routing
by the model name. An agent calls :func:`resolve_gateway` to get the litellm settings and maps them
onto its own surface (Python kwargs, an opencode provider block, or a CLI's
``-c model.model_kwargs.…``).

The typed ``provider`` union selects one of three routes:

* ``direct`` — no gateway; the agent calls the model provider.
* ``gateway`` — an explicit OpenAI-compatible endpoint in ``provider.extra_args``.
* ``internal`` — a named agent/deployment provider backed by the existing gateway environment.

With neither, the agent talks to the model provider directly on litellm's own defaults (bring your
own first-party ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` via ``forward_env``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from beagle.agents.core.provider import GatewayProvider, InternalProvider, provider_config

#: Direct API host per provider — for allowlisting the provider on a network-restricted benchmark
#: when NO gateway is configured. By litellm-style ``provider/…`` prefix, and by bare model-name
#: prefix. Best-effort; extend as needed. An unknown model yields ``None`` (litellm's own default).
_PROVIDER_HOST = {
    "openai": "api.openai.com", "anthropic": "api.anthropic.com",
    "gemini": "generativelanguage.googleapis.com", "google": "generativelanguage.googleapis.com",
    "mistral": "api.mistral.ai", "groq": "api.groq.com", "xai": "api.x.ai",
}
_MODEL_PREFIX_HOST: tuple[tuple[str, str], ...] = (
    ("gpt", "api.openai.com"), ("o1", "api.openai.com"), ("o3", "api.openai.com"),
    ("o4", "api.openai.com"), ("chatgpt", "api.openai.com"),
    ("claude", "api.anthropic.com"),
    ("gemini", "generativelanguage.googleapis.com"),
    ("mistral", "api.mistral.ai"), ("magistral", "api.mistral.ai"),
    ("grok", "api.x.ai"),
)


def provider_api_host(model: str) -> str | None:
    """Best-effort direct API host for ``model``'s provider — used to allowlist the provider on a
    network-restricted run when no gateway routes the call. Honors an explicit litellm
    ``provider/model`` prefix first, else the bare model-name prefix. ``None`` for an unknown model
    (the caller then leaves it to litellm's default; unrestricted benchmarks are unaffected)."""
    m = (model or "").strip().lower()
    if not m:
        return None
    if "/" in m:
        prov = m.split("/", 1)[0]
        if prov in _PROVIDER_HOST:
            return _PROVIDER_HOST[prov]
    for prefix, host in _MODEL_PREFIX_HOST:
        if m.startswith(prefix):
            return host
    return None


def gateway_key_pool() -> list[str]:
    """The ordered, de-duped gateway API-key pool: the singular ``LLM_GATEWAY_EXPRESS_API_KEY``
    first (explicit), then the ``…_LIST`` entries; blanks (stray commas) skipped. Empty when no
    key is configured. Callers pick ``[0]`` by default, or probe the pool to skip a blocked key."""
    single = (os.environ.get("LLM_GATEWAY_EXPRESS_API_KEY") or "").strip()
    listed = [k.strip() for k in (os.environ.get("LLM_GATEWAY_EXPRESS_API_KEY_LIST") or "")
              .replace(";", ",").split(",") if k.strip()]
    out: list[str] = []
    for k in ([single] if single else []) + listed:
        if k not in out:
            out.append(k)
    return out


def gateway_litellm_kwargs() -> dict[str, str] | None:
    """LiteLLM ``model_kwargs`` to route at the configured LLM Gateway Express, or ``None`` when no
    gateway is set (litellm uses its own provider defaults).

    Model-agnostic: ``api_base`` is litellm's provider-neutral endpoint (one URL for any model),
    the ``api_key`` comes from the forwarded key pool, and ``custom_llm_provider="openai"`` forces
    the OpenAI wire shape the unified proxy speaks — so gpt-5.5 or any claude route through it
    unchanged (the gateway routes by model name).

    Key selection is the FIRST NON-EMPTY entry of the pool (singular var, else the first usable
    ``…_LIST`` entry — blanks from stray commas skipped). NOTE: this picks a key once, up front —
    beagle can't rotate on a mid-run 401 because the agent (e.g. mini) owns its own LLM calls
    inside the container, and the gateway's 200/401 split can differ per replica / per endpoint
    (``/v1/chat/completions`` vs ``/v1/responses``) and even host-vs-container. A key that flips to
    "blocked" mid-rollout is a key-pool concern, not something this selection can recover."""
    url = (os.environ.get("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL") or "").strip()
    if not url:
        return None
    pool = gateway_key_pool()
    key = pool[0] if pool else "sk-noauth"
    return {"api_base": url, "api_key": key, "custom_llm_provider": "openai"}


def gateway_block(cfg: Mapping[str, Any] | None) -> dict[str, str] | None:
    """Validated explicit gateway arguments, or ``None`` for another provider type."""
    route = provider_config(dict(cfg or {}))
    if not isinstance(route, GatewayProvider):
        return None
    return route.extra_args.model_dump()


def config_gateway_kwargs(cfg: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """LiteLLM settings for the config-declared gateway (:func:`gateway_block`) — ``None`` when the
    config declares none.

    Same model-agnostic shape as :func:`gateway_litellm_kwargs`: ``api_base`` + ``api_key`` +
    ``custom_llm_provider="openai"`` (the OpenAI wire shape an org proxy speaks, so the model name
    alone picks the route). The key is read from the environment by the NAME the config gives
    (``api_key_env``) — the secret never enters a config file, and the run record keeps the variable
    name, so a run stays reproducible without leaking.

    ``auth_header`` covers a proxy that authenticates on its own header (e.g. ``x-api-key``) instead
    of ``Authorization: Bearer``: the key is then ALSO sent as ``extra_headers`` — litellm forwards
    those verbatim — while the standard bearer stays put, so a gateway accepting either works.

    An unset ``api_key_env`` yields the ``sk-noauth`` placeholder rather than an exception: the same
    best-effort contract as ``forward_env`` (``beagle evaluate --dry-run`` reports the missing var in
    its pre-flight, so it surfaces before spend). With no key resolved the custom header is OMITTED
    rather than sent empty — an otherwise-anonymous gateway must not be handed a blank credential
    header and reject the call for it.
    """
    route = provider_config(dict(cfg or {}))
    if not isinstance(route, GatewayProvider):
        return None
    block = route.extra_args.model_dump()
    key = (os.environ.get(block["api_key_env"]) or "").strip() if block["api_key_env"] else ""
    kwargs: dict[str, Any] = {"api_base": block["api_base"],
                              "api_key": key or "sk-noauth",
                              "custom_llm_provider": "openai"}
    if block["auth_header"] and key:
        kwargs["extra_headers"] = {block["auth_header"]: key}
    return kwargs


def resolve_gateway(cfg: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Resolve the selected provider route into gateway kwargs when applicable."""
    route = provider_config(dict(cfg or {}))
    if isinstance(route, GatewayProvider):
        return config_gateway_kwargs(dict(cfg or {}))
    if isinstance(route, InternalProvider):
        gateway = gateway_litellm_kwargs()
        if gateway is None:
            raise ValueError(
                f"internal provider {route.name!r} requires "
                "LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL to be set")
        return gateway
    return None


__all__ = ["config_gateway_kwargs", "gateway_block", "gateway_key_pool",
           "gateway_litellm_kwargs", "provider_api_host", "resolve_gateway"]
