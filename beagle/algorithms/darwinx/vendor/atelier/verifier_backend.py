"""OpenAI-compatible backend for ``atelier.verifier``.

Implements the ``VerifierBackend`` protocol against any OpenAI-compatible
chat-completions endpoint (real OpenAI, SFR Gateway, or any local
``openai`` API).

The backend uses the chat completions API's ``logprobs=True, top_logprobs=K``
options to retrieve per-token log-probability distributions. Atelier's
``logprobs_to_distribution`` then normalizes those into a probability
distribution over the score tokens (digits 1–9 by default).

# Model choice

The verifier should use a model **different from the proposer's family**
to avoid the failure mode where the proposer learns to produce
trajectories the verifier (= same family) happens to prefer. Suggested
defaults:

- Proposer = Claude Opus 4.6 (via cursor-agent) → Verifier = GPT-5.4-mini
  or Gemini-3.1-Pro.
- Proposer = GPT-5 family → Verifier = Claude (NOT supported here: Claude
  does not expose logprobs).

Claude is NOT supported as a verifier backend because Anthropic's API
does not return logprobs as of the time of writing.

# Cost considerations

For one trajectory at the default config (5 criteria × 3 repeats = 15
calls):
- Output is exactly 1 token → output cost is negligible.
- Input is ~10K tokens (transcript) per call → cost is dominated by input.

Practical tips:
- Prefer a small / mini model for verification (cost↓).
- Truncate transcripts before passing in (the verifier doesn't need every
  tool call's full output — head + tail of long blocks is usually enough).
- OpenAI prompt caching kicks in when the same prefix is sent repeatedly,
  so K repeats of the same criterion benefit from cache hits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .verifier import (
    DEFAULT_SCORE_TOKENS,
    ScoreDistribution,
    logprobs_to_distribution,
)


# ─── Defaults ─────────────────────────────────────────────────────────────


DEFAULT_TEMPERATURE = 0.7
"""K-repeat samples need variance, so we don't use temperature=0. 0.7 is
the value the LLM-as-a-Verifier paper reports as their working point."""

DEFAULT_TOP_LOGPROBS = 5
"""How many candidate tokens the chat-completions API should return
logprobs for. The SFR Gateway (and current OpenAI gpt-5.x models)
cap this at 5; older code defaulted to 20. The verifier converts
whatever it gets via ``logprobs_to_distribution`` so the smaller
window degrades smoothly — score tokens not in the top 5 just
get treated as ~zero probability (which is approximately true
for any well-anchored prompt)."""

DEFAULT_MAX_TOKENS = 16
"""Verifier responses use only the first content token's logprobs, but
gpt-5.x reasoning models can emit a few invisible reasoning tokens
before the visible answer. Allowing 16 leaves headroom; the score
parser still only inspects ``choice.logprobs.content[0]``."""


# ─── Config ───────────────────────────────────────────────────────────────


@dataclass
class OpenAIVerifierConfig:
    """Configuration for an OpenAI-compatible verifier backend.

    Required fields are pulled from environment variables (typically loaded
    via ``monet_eval.core.env.ensure_loaded()``) when ``api_key`` or
    ``base_url`` are left None.
    """

    model: str
    """Model identifier (e.g., 'gpt-5.4-mini', 'gpt-4o')."""

    api_key: str | None = None
    """If None, read from ``OPENAI_API_KEY`` (or ``OPENAI_GATEWAY_API_KEY``
    when ``base_url`` looks like the SFR Gateway)."""

    base_url: str | None = None
    """If None, use OpenAI's default endpoint."""

    temperature: float = DEFAULT_TEMPERATURE
    top_logprobs: int = DEFAULT_TOP_LOGPROBS
    max_tokens: int = DEFAULT_MAX_TOKENS

    request_timeout: float = 60.0
    """Per-call timeout in seconds."""

    extra_create_kwargs: dict[str, Any] = field(default_factory=dict)
    """Forwarded to ``client.chat.completions.create``. Useful for
    ``response_format``, custom headers, etc."""

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = _resolve_api_key(self.base_url)
        if self.api_key is None:
            raise RuntimeError(
                "OpenAIVerifierConfig: api_key not provided and not in env. "
                "Set OPENAI_API_KEY (or OPENAI_GATEWAY_API_KEY for SFR Gateway)."
            )


def _resolve_api_key(base_url: str | None) -> str | None:
    """Pick the right env-var for the configured base_url."""
    if base_url and "gateway.salesforce" in base_url:
        return os.environ.get("OPENAI_GATEWAY_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
    return os.environ.get("OPENAI_API_KEY")


# ─── Project-wide credentials adapter ────────────────────────────────────


def config_from_credentials(
    *,
    model: str,
    provider: str = "sfr_gateway",
    temperature: float = DEFAULT_TEMPERATURE,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_timeout: float = 60.0,
    extra_create_kwargs: dict[str, Any] | None = None,
) -> OpenAIVerifierConfig:
    """Build an ``OpenAIVerifierConfig`` from ``monet_eval.core.credentials``.

    This is the project-canonical entry point: it loads the ``.env`` file,
    validates the required env vars for ``provider``, and uses
    ``core.credentials._SFR_GATEWAY_DEFAULT_BASE_URL`` as the fallback
    base URL for the SFR Gateway profile (matching what ``monet_code``
    itself does).

    Use this in production code. ``OpenAIVerifierConfig(...)`` directly is
    fine for tests where you want to inject hand-built credentials.

    Raises ``MissingCredentialError`` if the required env vars are unset.
    """
    # Lazy import — core.credentials is a small module but importing it
    # eagerly creates a hard dep on monet_eval.core from the atelier
    # package, which we want to keep as a soft boundary so atelier can
    # be unit-tested in isolation.
    # coding-bench has no ``monet_eval.core.credentials``; resolve from the
    # process environment (the launch exports the provider credentials).
    import os

    env = dict(os.environ)

    if provider == "sfr_gateway":
        api_key = env["OPENAI_GATEWAY_API_KEY"]
    elif provider == "openai":
        api_key = env["OPENAI_API_KEY"]
    else:
        raise ValueError(
            f"verifier backend only supports openai/sfr_gateway, got {provider!r}"
        )

    base_url = env.get("OPENAI_BASE_URL")

    return OpenAIVerifierConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        top_logprobs=top_logprobs,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        extra_create_kwargs=extra_create_kwargs or {},
    )


def backend_from_credentials(
    *,
    model: str,
    provider: str = "sfr_gateway",
    **kwargs: Any,
) -> "OpenAIVerifierBackend":
    """Shorthand: build a fully-wired ``OpenAIVerifierBackend`` from the
    project credentials. See ``config_from_credentials``."""
    config = config_from_credentials(model=model, provider=provider, **kwargs)
    return OpenAIVerifierBackend(config)


# ─── Backend implementation ──────────────────────────────────────────────


class OpenAIVerifierBackend:
    """Chat-completions logprob backend for ``atelier.verifier.Verifier``.

    Build with a config; call ``.score(prompt=, score_tokens=)`` to get
    a ScoreDistribution.

    Lazy-imports the ``openai`` package so the atelier module can be
    imported without that dependency available (e.g., during unit tests
    that swap in a fake backend).
    """

    def __init__(self, config: OpenAIVerifierConfig):
        self.config = config
        self._client = None  # built lazily

    def _get_client(self):
        if self._client is None:
            # Lazy import so importing atelier.verifier_backend doesn't
            # require openai installed when callers only need the config
            # dataclass (e.g., for serialization).
            try:
                import openai  # noqa: F401  (used via client cls below)
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "atelier.verifier_backend requires the openai client. "
                    "Install via the atelier extras: "
                    "`uv sync --extra atelier` (or `pip install openai`)."
                ) from e

            kwargs: dict[str, Any] = {
                "api_key": self.config.api_key,
                "timeout": self.config.request_timeout,
            }
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            # SFR Gateway uses x-api-key, not Authorization: Bearer.
            # See atelier.matchfix_gate.OpenAIChatBackend for the
            # same fix + the failure mode it addresses.
            if self.config.base_url and "gateway.salesforce" in self.config.base_url:
                kwargs["default_headers"] = {"x-api-key": self.config.api_key}
            self._client = OpenAI(**kwargs)
        return self._client

    def score(
        self, *, prompt: str, score_tokens: tuple[str, ...] = DEFAULT_SCORE_TOKENS
    ) -> ScoreDistribution:
        """Send the prompt; parse the top-K logprobs into a distribution
        over the score tokens.

        The chat call is sent as a single user message; we don't include
        a system message because the prompt template already carries the
        role + instructions ("You are an expert evaluator …").
        """
        client = self._get_client()
        # gpt-5.x models on SFR Gateway require max_completion_tokens
        # instead of max_tokens (see matchfix_gate.OpenAIChatBackend).
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_tokens,
            logprobs=True,
            top_logprobs=self.config.top_logprobs,
            **self.config.extra_create_kwargs,
        )

        # Defensive parse — the API has been known to return objects with
        # missing logprob blocks on rare error paths.
        choice = response.choices[0]
        if choice.logprobs is None or not choice.logprobs.content:
            # No logprob block — fall back to uniform.
            return logprobs_to_distribution(
                {}, score_tokens=score_tokens
            )

        # We capped at 1 output token, so choice.logprobs.content has
        # exactly one entry. Its top_logprobs is the list we need.
        first_token = choice.logprobs.content[0]
        top = first_token.top_logprobs or []

        logprob_map: dict[str, float] = {}
        # The top-logprob entries are TopLogprob objects with `.token` and
        # `.logprob`. Tokens may carry leading whitespace (e.g., " 7"); we
        # strip that to match our digit score_tokens.
        for entry in top:
            token = (getattr(entry, "token", None) or "").strip()
            if not token:
                continue
            logprob = float(getattr(entry, "logprob", float("-inf")))
            # If the same digit appears multiple times (very rare, but
            # could happen across whitespace variants after stripping),
            # keep the higher (less negative) logprob.
            if token not in logprob_map or logprob > logprob_map[token]:
                logprob_map[token] = logprob

        return logprobs_to_distribution(
            logprob_map, score_tokens=score_tokens
        )


__all__ = [
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_LOGPROBS",
    "DEFAULT_MAX_TOKENS",
    "OpenAIVerifierConfig",
    "OpenAIVerifierBackend",
    "config_from_credentials",
    "backend_from_credentials",
]
