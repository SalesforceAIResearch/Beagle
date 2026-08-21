"""The benchmark cache-root guard (renamed 2026-07-31: XRLENV_HARBOR_CACHE ->
XRLENV_BENCHMARK_CACHE, .../xrlenv_harbor_cache -> .../xrlenv_benchmark_cache). The retired
var/path must be HARD-REJECTED so a downstream user on the stale cache fails loud."""
from __future__ import annotations

import pytest
from xrlenv_plugins.benchmarks._benchmark_cache import (
    benchmark_cache_root,
    guard_legacy_cache_env,
)

_NEW = "/path/to/benchmark-cache"
_OLD = "/path/to/data/xrlenv_harbor_cache"


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XRLENV_HARBOR_CACHE", raising=False)
    monkeypatch.delenv("XRLENV_BENCHMARK_CACHE", raising=False)


def test_guard_rejects_legacy_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XRLENV_HARBOR_CACHE", _OLD)
    with pytest.raises(SystemExit, match="XRLENV_HARBOR_CACHE is retired"):
        guard_legacy_cache_env()


def test_guard_rejects_empty_legacy_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    # audit Low: the rule is "must not be SET" — an explicitly empty value is a stale-
    # migration signal too, so membership (not truthiness) must reject it.
    _clear(monkeypatch)
    monkeypatch.setenv("XRLENV_HARBOR_CACHE", "")
    with pytest.raises(SystemExit, match="XRLENV_HARBOR_CACHE is retired"):
        guard_legacy_cache_env()


def test_guard_rejects_legacy_env_var_even_with_new_var_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # both set -> still reject: the stale legacy var must be removed.
    _clear(monkeypatch)
    monkeypatch.setenv("XRLENV_HARBOR_CACHE", _OLD)
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", _NEW)
    with pytest.raises(SystemExit, match="retired"):
        guard_legacy_cache_env()


def test_guard_rejects_legacy_path_in_new_var(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", _OLD)   # new var but OLD path
    with pytest.raises(SystemExit, match="retired"):
        guard_legacy_cache_env()


def test_guard_rejects_legacy_path_in_explicit_dest(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    with pytest.raises(SystemExit, match="retired"):
        guard_legacy_cache_env(_OLD)


def test_guard_passes_on_new_var_and_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", _NEW)
    guard_legacy_cache_env()          # no raise
    guard_legacy_cache_env(_NEW)      # no raise


def test_benchmark_cache_root_resolves_new_var(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", _NEW)
    assert benchmark_cache_root() == _NEW
    assert benchmark_cache_root("/some/explicit") == "/some/explicit"


def test_benchmark_cache_root_unset_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    with pytest.raises(SystemExit, match="no cache root"):
        benchmark_cache_root()
