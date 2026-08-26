"""Optional LLM client + the QA ``ask`` mode.

The client is a thin OpenAI-compatible ``/chat/completions`` caller (stdlib
``urllib``) that satisfies the :class:`~trace_analyzer.proposers.LLMClient`
protocol, so the same object powers LLM proposers, LLM filters, and ``ask``.
It is entirely optional: the deterministic rule proposers run without it, and it
fails loud the moment it can't resolve an endpoint + key.

``ask`` is QA over a trace. To blunt the "context rot" the blog warns about on
long traces, it map-reduces: pull question-relevant evidence per chunk, then
answer from the gathered evidence (small traces answer in one shot).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .chunks import TraceView
from .model import CanonicalTrajectory

# Known gateways in priority order, each a (base_url_env, key_envs) PAIR. The key
# is taken from the *same* gateway as the URL, so an unrelated/stale key (e.g. an
# OPENAI_API_KEY left in the env) can't shadow the one that matches the chosen
# URL — exactly the Express case, where the valid key is LLM_GATEWAY_EXPRESS_API_KEY.
# `ask`/proposers POST to ``<base>/chat/completions``; that bare path is what the
# Express forwarder whitelists (no ``/v1``), and SFR's base already carries ``/v1``.
_GATEWAYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("DARWINX_TRACE_BASE_URL", ("DARWINX_TRACE_API_KEY",)),
    ("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", ("LLM_GATEWAY_EXPRESS_API_KEY",)),
    ("OPENAI_BASE_URL", ("OPENAI_API_KEY", "OPENAI_GATEWAY_API_KEY")),
    ("SFR_GATEWAY_OPENAI_URL", ("X_API_KEY", "OPENAI_GATEWAY_API_KEY")),
)
_BASE_URL_ENV = tuple(g[0] for g in _GATEWAYS)
_KEY_ENV = tuple(dict.fromkeys(k for _, ks in _GATEWAYS for k in ks))
_WHOLE_TRACE_BUDGET = 40_000  # chars; below this, answer in one shot


class LLMError(Exception):
    pass


def _env(name: str) -> str | None:
    return os.environ.get(name) or None


def _resolve_gateway(base_url: str | None, api_key: str | None) -> tuple[str | None, str | None]:
    """Resolve (base_url, key) as a matched pair from the configured gateways."""
    if base_url:  # explicit URL: use given key, else any available key
        return base_url, api_key or next((_env(k) for k in _KEY_ENV if _env(k)), None)
    for url_env, key_envs in _GATEWAYS:
        url = _env(url_env)
        if url:
            return url, api_key or next((_env(k) for k in key_envs if _env(k)), None)
    return None, api_key


class OpenAIClient:
    """Minimal OpenAI-compatible chat client (implements ``LLMClient``)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        base, key = _resolve_gateway(base_url, api_key)
        if not base:
            raise LLMError("no base URL; pass --base-url or set one of: " + ", ".join(_BASE_URL_ENV))
        if not key:
            raise LLMError("no API key; pass --api-key or set one of: " + ", ".join(_KEY_ENV))
        self.base_url: str = base
        self.api_key: str = key
        self.model = model or os.environ.get("DARWINX_TRACE_MODEL") or "gpt-5.5"
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        # Resilient send: retry on 429 (rate limit) and transient 5xx with
        # exponential backoff + jitter, so a request-rate burst doesn't get the
        # chunk SKIPPED (which silently degrades the LLM gradient to the fragile
        # rule-only proposers). Tunable via DARWINX_TRACE_LLM_MAX_RETRIES /
        # DARWINX_TRACE_LLM_BACKOFF_S.
        import random as _random
        import time as _time
        max_retries = int(os.environ.get("DARWINX_TRACE_LLM_MAX_RETRIES", "5") or 5)
        backoff = float(os.environ.get("DARWINX_TRACE_LLM_BACKOFF_S", "10") or 10)
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                code = exc.code
                detail = exc.read().decode("utf-8", "replace")[:300]
                last_exc = LLMError(f"LLM request failed ({code}): {detail}")
                if code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    # Honor Retry-After when present, else exponential backoff.
                    ra = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        delay = float(ra) if ra else backoff * (2 ** attempt)
                    except (TypeError, ValueError):
                        delay = backoff * (2 ** attempt)
                    delay = min(delay, 90.0) + _random.uniform(0, backoff)
                    _time.sleep(delay)
                    continue
                raise last_exc from exc
            except urllib.error.URLError as exc:
                last_exc = LLMError(f"LLM request failed: {exc.reason}")
                if attempt < max_retries:
                    _time.sleep(backoff * (2 ** attempt) + _random.uniform(0, backoff))
                    continue
                raise last_exc from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected LLM response shape: {body}") from exc


_QA_SYSTEM = (
    "You are an agent-trajectory analyst. Answer the question precisely using only "
    "the trace evidence given. Cite evidence as [trace_id #message_index]. If the "
    "trace does not show something, say so."
)


def ask(
    trajectories: list[CanonicalTrajectory],
    question: str,
    *,
    client: OpenAIClient | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    client = client or OpenAIClient(model, base_url, api_key)
    blocks = []
    for traj in trajectories:
        view = TraceView.build(traj)
        rendered = "\n".join(c.text for c in view.chunks(size=10_000))  # one big render
        header = f"### trace_id: {traj.trace_id} (terminal={traj.terminal})"
        if len(rendered) <= _WHOLE_TRACE_BUDGET:
            blocks.append(f"{header}\n{rendered}")
        else:
            blocks.append(f"{header}\n{_mapreduce_evidence(view, question, client)}")
    prompt = "\n\n".join(blocks) + f"\n\n---\nQuestion: {question}"
    return client.complete(_QA_SYSTEM, prompt)


def _mapreduce_evidence(view: TraceView, question: str, client: OpenAIClient) -> str:
    """Pull question-relevant lines from each chunk (map) for a long trace."""
    system = (
        "Extract only lines from this trace chunk relevant to the question, each "
        "prefixed with its [#message_index]. If nothing is relevant, reply 'none'."
    )
    kept = []
    for chunk in view.chunks(size=12):
        got = client.complete(system, f"Question: {question}\n\n{chunk.text}").strip()
        if got and got.lower() != "none":
            kept.append(got)
    return "\n".join(kept) if kept else "(no relevant evidence found)"
