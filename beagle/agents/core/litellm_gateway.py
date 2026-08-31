"""Reach the LLM Gateway Express from a **LiteLLM-backed** agent — shared infrastructure.

LiteLLM is the model layer many agents drive (mini-swe today, more later), so *how* to point it at
the gateway lives HERE, once — not re-implemented per agent. The gateway is a **unified
OpenAI-compatible proxy** (``scripts/gateway/gateway_proxy.py``): one
``LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL`` serves any model — gpt *or* claude — over the OpenAI wire
shape, routing by the model name. An agent forwards the gateway creds (``forward_env``) and calls
:func:`gateway_litellm_kwargs` to get the litellm settings; it maps them onto its own surface
(Python kwargs, or a CLI's ``-c model.model_kwargs.…``).
"""

from __future__ import annotations

import os

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


__all__ = ["gateway_litellm_kwargs", "gateway_key_pool", "provider_api_host"]
