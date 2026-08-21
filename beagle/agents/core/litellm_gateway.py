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


def gateway_litellm_kwargs() -> dict[str, str] | None:
    """LiteLLM ``model_kwargs`` to route at the configured LLM Gateway Express, or ``None`` when no
    gateway is set (litellm uses its own provider defaults).

    Model-agnostic: ``api_base`` is litellm's provider-neutral endpoint (one URL for any model),
    the ``api_key`` comes from the forwarded key pool, and ``custom_llm_provider="openai"`` forces
    the OpenAI wire shape the unified proxy speaks — so gpt-5.5 or any claude route through it
    unchanged (the gateway routes by model name)."""
    url = (os.environ.get("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL") or "").strip()
    if not url:
        return None
    key = (os.environ.get("LLM_GATEWAY_EXPRESS_API_KEY")
           or (os.environ.get("LLM_GATEWAY_EXPRESS_API_KEY_LIST") or "")
           .replace(";", ",").split(",")[0].strip()
           or "sk-noauth")
    return {"api_base": url, "api_key": key, "custom_llm_provider": "openai"}


__all__ = ["gateway_litellm_kwargs"]
