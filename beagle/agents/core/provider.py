"""Typed LLM-provider routing shared by every agent adapter.

``provider`` answers *where/how the agent reaches its model*.  It is deliberately
independent of ``forward_env``, which remains generic host-to-container plumbing.
The public config is a discriminated union::

    provider: {type: direct, name: openai}
    provider:
      type: gateway
      name: my-gateway
      extra_args: {api_base: https://gateway.example/v1, api_key_env: MY_KEY}
    provider: {type: internal, name: deployment-provider}

Adapters consume the same parsed object for invocation and network allowlisting,
so those two views cannot silently disagree.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class _ProviderBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = ""

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        return value.strip()


class DirectProvider(_ProviderBase):
    """Call a first-party model API, inferred from the model when ``name`` is empty."""

    type: Literal["direct"] = "direct"
    extra_args: dict[str, Any] = Field(default_factory=dict)


class GatewayArgs(BaseModel):
    """Arguments for an explicit OpenAI-compatible gateway."""

    model_config = ConfigDict(extra="forbid", strict=True)

    api_base: str
    api_key_env: str = ""
    auth_header: str = ""

    @field_validator("api_base", "api_key_env", "auth_header")
    @classmethod
    def _strip_string(cls, value: str) -> str:
        return value.strip()

    @field_validator("api_base")
    @classmethod
    def _require_api_base(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value


class GatewayProvider(_ProviderBase):
    """Call an explicitly configured OpenAI-compatible endpoint."""

    type: Literal["gateway"]
    name: str
    extra_args: GatewayArgs

    @field_validator("name")
    @classmethod
    def _require_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty for provider type 'gateway'")
        return value


class InternalProvider(_ProviderBase):
    """Use an agent/deployment-native named provider."""

    type: Literal["internal"]
    name: str
    extra_args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _require_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty for provider type 'internal'")
        return value


Provider = Annotated[
    DirectProvider | GatewayProvider | InternalProvider,
    Field(discriminator="type"),
]
_PROVIDER_ADAPTER = TypeAdapter(Provider)


def provider_config(cfg: dict[str, Any] | None) -> Provider:
    """Parse ``cfg["provider"]``; absence means direct model-provider access.

    This is a hard cutover.  The former scalar ``provider`` and sibling ``gateway``
    shapes fail with migration guidance instead of being guessed.
    """
    cfg = cfg or {}
    if "gateway" in cfg:
        raise ValueError(
            "top-level `gateway:` was replaced by `provider: {type: gateway, "
            "name: <gateway-name>, extra_args: {api_base, api_key_env, auth_header}}`"
        )
    raw = cfg.get("provider")
    if raw is None:
        return DirectProvider()
    if not isinstance(raw, dict):
        raise ValueError(  # noqa: TRY004 — config migration errors are uniformly ValueError
            "scalar `provider:` was replaced by a mapping, e.g. "
            "`provider: {type: internal, name: <provider-name>}`"
        )
    try:
        return _PROVIDER_ADAPTER.validate_python(raw)
    except ValueError as exc:
        raise ValueError(f"invalid provider configuration: {exc}") from exc


def provider_dict(provider: Provider) -> dict[str, Any]:
    """JSON-compatible representation stored in ``AgentSpec.config``."""
    data = provider.model_dump(mode="json", exclude_defaults=True, exclude={"type"})
    return {"type": provider.type, **data}


__all__ = [
    "DirectProvider",
    "GatewayArgs",
    "GatewayProvider",
    "InternalProvider",
    "Provider",
    "provider_config",
    "provider_dict",
]
