"""Shared litellm → OpenAI-compatible gateway routing — reusable by any litellm-backed agent."""

from __future__ import annotations

import pytest

from beagle.agents.core.litellm_gateway import (
    config_gateway_kwargs,
    gateway_block,
    gateway_key_pool,
    gateway_litellm_kwargs,
    resolve_gateway,
)
from beagle.agents.core.provider import (
    DirectProvider,
    GatewayProvider,
    InternalProvider,
    provider_config,
    provider_dict,
)


def test_key_pool_is_ordered_and_deduped(monkeypatch) -> None:
    # The probe iterates this pool: singular var first (explicit), then the LIST, blanks skipped,
    # order-preserving dedup.
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_API_KEY", "sk-single")
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_API_KEY_LIST", "sk-single, k2 ,, k3;k2")
    assert gateway_key_pool() == ["sk-single", "k2", "k3"]


def test_provider_api_host_maps_model_to_its_provider() -> None:
    from beagle.agents.core.litellm_gateway import provider_api_host
    assert provider_api_host("gpt-5.5") == "api.openai.com"
    assert provider_api_host("o1-mini") == "api.openai.com"                 # reasoning-model prefixes
    assert provider_api_host("o3-mini") == "api.openai.com"
    assert provider_api_host("o4-preview") == "api.openai.com"
    assert provider_api_host("claude-sonnet-4-5") == "api.anthropic.com"
    assert provider_api_host("anthropic/claude-x") == "api.anthropic.com"   # explicit provider prefix
    assert provider_api_host("gemini-2.5-pro") == "generativelanguage.googleapis.com"
    assert provider_api_host("mystery-9") is None                           # unknown → litellm default
    assert provider_api_host("") is None


def test_none_when_no_gateway(monkeypatch) -> None:
    monkeypatch.delenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", raising=False)
    assert gateway_litellm_kwargs() is None


def test_kwargs_when_gateway_set(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://node:18088/")
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_API_KEY", "sk-real")
    # provider-neutral api_base + a key + force the OpenAI wire shape (the unified proxy's shape)
    assert gateway_litellm_kwargs() == {
        "api_base": "http://node:18088/", "api_key": "sk-real", "custom_llm_provider": "openai"}


def test_key_falls_back_to_first_of_the_list(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://x/")
    monkeypatch.delenv("LLM_GATEWAY_EXPRESS_API_KEY", raising=False)
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_API_KEY_LIST", "k1, k2 ,k3")   # proxy round-robins these
    assert gateway_litellm_kwargs()["api_key"] == "k1"


def test_key_skips_blank_entries_in_the_list(monkeypatch) -> None:
    # A stray leading comma / empty entry must not yield an empty api_key — pick the first usable key.
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://x/")
    monkeypatch.delenv("LLM_GATEWAY_EXPRESS_API_KEY", raising=False)
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_API_KEY_LIST", " , ,k2,k3")
    assert gateway_litellm_kwargs()["api_key"] == "k2"


def test_noauth_when_pool_is_empty(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://x/")
    monkeypatch.delenv("LLM_GATEWAY_EXPRESS_API_KEY", raising=False)
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_API_KEY_LIST", " , , ")   # all blank
    assert gateway_litellm_kwargs()["api_key"] == "sk-noauth"


# -- typed provider routes -------------------------------------------------------------------


def _gateway(**extra_args):
    return {"provider": {"type": "gateway", "name": "org", "extra_args": extra_args}}


def _internal(name: str = "deployment-gateway"):
    return {"provider": {"type": "internal", "name": name}}


def test_provider_union_parses_and_serializes() -> None:
    assert isinstance(provider_config({}), DirectProvider)
    assert isinstance(provider_config(_gateway(api_base="https://gw/v1")), GatewayProvider)
    assert isinstance(provider_config(_internal()), InternalProvider)
    assert provider_dict(provider_config(_internal())) == {
        "type": "internal", "name": "deployment-gateway"}


def test_config_gateway_reads_the_key_from_the_named_env_var(monkeypatch) -> None:
    monkeypatch.setenv("MY_ORG_API_KEY", "sk-org")
    kw = config_gateway_kwargs(_gateway(
        api_base="https://gw.example/openai/v1", api_key_env="MY_ORG_API_KEY"))
    assert kw == {"api_base": "https://gw.example/openai/v1", "api_key": "sk-org",
                  "custom_llm_provider": "openai"}


def test_config_gateway_adds_a_custom_auth_header(monkeypatch) -> None:
    monkeypatch.setenv("X_API_KEY", "sk-org")
    kw = config_gateway_kwargs(_gateway(
        api_base="https://gw.example/openai/v1", api_key_env="X_API_KEY",
        auth_header="x-api-key"))
    assert kw["api_key"] == "sk-org" and kw["extra_headers"] == {"x-api-key": "sk-org"}


def test_config_gateway_key_is_best_effort_when_the_var_is_unset(monkeypatch) -> None:
    monkeypatch.delenv("MY_ORG_API_KEY", raising=False)
    kw = config_gateway_kwargs(_gateway(
        api_base="https://gw.example/v1", api_key_env="MY_ORG_API_KEY"))
    assert kw["api_key"] == "sk-noauth"


@pytest.mark.parametrize("provider", [
    "openai",
    {"type": "gateway", "extra_args": {}},
    {"type": "gateway", "name": "", "extra_args": {"api_base": "https://gw/v1"}},
    {"type": "gateway", "name": "org", "extra_args": {"api_base": ["https://gw/v1"]}},
    {"type": "gateway", "name": "org", "extra_args": {
        "api_base": "https://gw/v1", "unknown": 1}},
    {"type": "internal"},
    {"type": "internal", "name": ""},
    {"type": "mystery"},
])
def test_malformed_provider_fails_loud(provider) -> None:
    with pytest.raises(ValueError, match="provider"):
        provider_config({"provider": provider})


def test_old_gateway_shape_has_focused_migration_error() -> None:
    with pytest.raises(ValueError, match="top-level `gateway:` was replaced"):
        provider_config({"gateway": {"api_base": "https://gw/v1"}})


def test_custom_header_is_omitted_when_no_key_resolved(monkeypatch) -> None:
    monkeypatch.delenv("MY_ORG_API_KEY", raising=False)
    kw = config_gateway_kwargs(_gateway(
        api_base="https://gw/v1", api_key_env="MY_ORG_API_KEY", auth_header="X-Api-Key"))
    assert "extra_headers" not in kw and kw["api_key"] == "sk-noauth"


def test_explicit_route_controls_env_gateway_precedence(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://node:18088/")
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_API_KEY", "sk-env")
    monkeypatch.setenv("MY_ORG_API_KEY", "sk-org")
    assert resolve_gateway({}) is None
    assert resolve_gateway({"provider": {"type": "direct"}}) is None
    assert resolve_gateway(_internal())["api_base"] == "http://node:18088/"
    kw = resolve_gateway(_gateway(
        api_base="https://gw.example/v1", api_key_env="MY_ORG_API_KEY"))
    assert kw["api_base"] == "https://gw.example/v1" and kw["api_key"] == "sk-org"


def test_internal_route_fails_when_deployment_endpoint_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", raising=False)
    with pytest.raises(ValueError, match="internal provider.*requires"):
        resolve_gateway(_internal())
