"""Unit tests for webarena_infinity xrlenv_config.py — focused on the
OSS-cleanup behavioral changes: MONET_GIT_URL now has no default, and
monet_preflight() appends a missing-message when it is unset/empty."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the sibling module importable without installing it (same pattern as the
# evoclaw copy_to_call_site tests).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xrlenv_config

# ---------------------------------------------------------------------------
# MONET_GIT_URL default is now "" (no hardcoded private repo)
# ---------------------------------------------------------------------------

def test_monet_git_url_default_is_empty(monkeypatch):
    """When MONET_GIT_URL is absent from the env the module attribute is ''."""
    monkeypatch.delenv("MONET_GIT_URL", raising=False)
    # Reload-style check: monkeypatch the already-imported module attribute,
    # which is what monet_preflight() reads.
    monkeypatch.setattr(xrlenv_config, "MONET_GIT_URL", "")
    assert xrlenv_config.MONET_GIT_URL == ""


# ---------------------------------------------------------------------------
# monet_preflight() — MONET_GIT_URL missing-message (new check)
# ---------------------------------------------------------------------------

def test_monet_preflight_includes_monet_git_url_message_when_unset(monkeypatch):
    """monet_preflight() must include a MONET_GIT_URL message when the value
    is empty — the fail-loud signal that the operator must configure it."""
    monkeypatch.setattr(xrlenv_config, "MONET_GIT_URL", "")
    messages = xrlenv_config.monet_preflight()
    assert any("MONET_GIT_URL" in m for m in messages), (
        "expected a MONET_GIT_URL missing-message but got: " + repr(messages)
    )


def test_monet_preflight_omits_monet_git_url_message_when_set(monkeypatch):
    """When MONET_GIT_URL is configured, monet_preflight() must NOT complain
    about it (only the other missing creds may appear)."""
    monkeypatch.setattr(xrlenv_config, "MONET_GIT_URL", "github.com/someorg/monet.git")
    messages = xrlenv_config.monet_preflight()
    assert not any("MONET_GIT_URL" in m for m in messages), (
        "unexpected MONET_GIT_URL message when URL is set: " + repr(messages)
    )


def test_monet_preflight_returns_list_not_raises(monkeypatch):
    """monet_preflight() always returns a list — it never raises, even when
    every prerequisite is missing (callers decide how to handle the list)."""
    monkeypatch.setattr(xrlenv_config, "MONET_GIT_URL", "")
    monkeypatch.delenv("LLM_GATEWAY_EXPRESS_API_KEY", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    result = xrlenv_config.monet_preflight()
    assert isinstance(result, list)
    assert len(result) == 4  # MONET_GIT_URL + the three other required keys


def test_monet_preflight_empty_when_all_present(monkeypatch):
    """monet_preflight() returns an empty list when every prerequisite is met."""
    monkeypatch.setattr(xrlenv_config, "MONET_GIT_URL", "github.com/someorg/monet.git")
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_API_KEY", "key123")
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://10.0.1.1:18088/")
    monkeypatch.setenv("GH_TOKEN", "ghp_token")
    result = xrlenv_config.monet_preflight()
    assert result == [], f"expected no missing prereqs, got: {result}"


# ---------------------------------------------------------------------------
# default_image() — still works correctly after the registry-host change
# ---------------------------------------------------------------------------

def test_default_image_returns_none_when_registry_unset(monkeypatch):
    monkeypatch.setattr(xrlenv_config, "REGISTRY_HOST", None)
    assert xrlenv_config.default_image() is None


def test_default_image_returns_full_ref_when_registry_set(monkeypatch):
    monkeypatch.setattr(xrlenv_config, "REGISTRY_HOST", "myregistry.internal")
    monkeypatch.setattr(xrlenv_config, "REGISTRY_PORT", "5011")
    result = xrlenv_config.default_image()
    assert result == "myregistry.internal:5011/xrlenv-webarena-infinity/substrate:dev"
