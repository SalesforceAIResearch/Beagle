"""Provider-neutral token accounting with a cache-status breakdown.

Agents' native usage formats disagree on cache semantics, so a uniform "add the cached
number" would be wrong:

* **opencode** reports ``cache.read`` / ``cache.write`` *in addition to* a fresh ``input``.
* **mini-swe** (OpenAI) reports ``prompt_tokens_details.cached_tokens`` as a *subset of*
  ``prompt_tokens``.
* **monet** reports a single ``cacheTokens`` (no read/write split).

Each agent's parser normalizes into :class:`Usage` — four **disjoint** buckets (fresh input,
cache-read, cache-write, output) — and :meth:`Usage.to_token_counts` renders the canonical
:data:`~beagle.types.TokenCounts` dict written to ``result.json`` / ``run.json``.

The legacy ``prompt`` / ``completion`` / ``total`` keys are preserved (``prompt`` == total
billable input), so nothing downstream breaks; the added ``input_uncached`` / ``cache_read`` /
``cache_write`` give the cache split a cost estimate needs (cost = Σ tokens · price, computed
downstream — beagle stays pricing-agnostic). Invariant::

    prompt = input_uncached + cache_read + cache_write
"""

from __future__ import annotations

from typing import NamedTuple


class Usage(NamedTuple):
    """One rollout's token usage, split by cache status (all four buckets disjoint)."""

    input_uncached: int = 0  #: fresh input — full input price
    cache_read: int = 0      #: cache-hit input — discounted price
    cache_write: int = 0     #: cache-creation input — premium price (0 where a provider doesn't bill it)
    output: int = 0          #: completion/output (includes provider "reasoning" tokens) — output price

    @property
    def input(self) -> int:
        """Total billable input = fresh + cache-read + cache-write."""
        return self.input_uncached + self.cache_read + self.cache_write

    def to_token_counts(self) -> dict[str, int]:
        """Render the canonical :data:`TokenCounts` dict (legacy keys + the cache breakdown)."""
        inp = self.input
        return {
            "prompt": inp,               # legacy: total billable input
            "completion": self.output,   # legacy
            "total": inp + self.output,  # legacy
            "input_uncached": self.input_uncached,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
        }


def add(a: Usage, b: Usage) -> Usage:
    """Field-wise sum (``+`` on a NamedTuple concatenates, so use this to accumulate)."""
    return Usage(a.input_uncached + b.input_uncached, a.cache_read + b.cache_read,
                 a.cache_write + b.cache_write, a.output + b.output)


__all__ = ["Usage", "add"]
