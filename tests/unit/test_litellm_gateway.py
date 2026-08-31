"""Shared litellm → LLM-Gateway-Express routing — reusable by any litellm-backed agent."""

from __future__ import annotations

from beagle.agents.core.litellm_gateway import gateway_key_pool, gateway_litellm_kwargs


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
