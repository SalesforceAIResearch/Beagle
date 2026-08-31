"""Unit tests for the provider-neutral :class:`Usage` token accounting (agents/core/usage.py).

The load-bearing invariant a downstream cost estimate relies on: ``prompt`` (total billable input)
== ``input_uncached + cache_read + cache_write``, with the four buckets disjoint."""

from __future__ import annotations

from beagle.agents.core.usage import Usage, add


def test_to_token_counts_holds_the_cache_invariant() -> None:
    u = Usage(input_uncached=800, cache_read=400, cache_write=34, output=567)
    tc = u.to_token_counts()
    assert tc == {"prompt": 1234, "completion": 567, "total": 1801,
                  "input_uncached": 800, "cache_read": 400, "cache_write": 34}
    # prompt == the disjoint input buckets; total == prompt + output.
    assert tc["prompt"] == tc["input_uncached"] + tc["cache_read"] + tc["cache_write"]
    assert tc["total"] == tc["prompt"] + tc["completion"]
    assert u.input == 1234


def test_empty_usage_is_all_zero() -> None:
    assert Usage() == Usage(0, 0, 0, 0)
    assert Usage().to_token_counts() == {"prompt": 0, "completion": 0, "total": 0,
                                         "input_uncached": 0, "cache_read": 0, "cache_write": 0}


def test_add_is_fieldwise() -> None:
    # `add` accumulates each bucket independently (plain `+` on a NamedTuple would concatenate).
    total = add(Usage(1, 2, 3, 4), Usage(10, 20, 30, 40))
    assert total == Usage(input_uncached=11, cache_read=22, cache_write=33, output=44)
