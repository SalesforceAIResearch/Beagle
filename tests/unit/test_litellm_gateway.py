"""Shared litellm → LLM-Gateway-Express routing — reusable by any litellm-backed agent."""

from __future__ import annotations

from beagle.agents.core.litellm_gateway import gateway_litellm_kwargs


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
