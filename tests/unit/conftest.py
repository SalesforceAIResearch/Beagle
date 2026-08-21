"""Unit-tier isolation: make the mechanical tests hermetic.

A developer's shell (or the repo ``.env``) commonly exports ``XRLENV_BENCHMARK_CACHE``
and the cluster/consumer tokens for their own dev workflow. Several unit tests
exercise code that falls back to those env vars (``HarborCache`` reads
``$XRLENV_BENCHMARK_CACHE``; the cluster runtime reads ``XRLENV_GRPC_*``). Without
stripping them, a unit test would pass on a clean checkout and change behavior the
moment the developer configured a cluster — the classic "works on CI, flakes
locally" split. Strip them at unit-session start so this tier is reproducible
regardless of shell/`.env` state.

This is scoped to ``tests/unit/`` on purpose: smoke tests (``tests/smoke/``) need
the live environment and must NOT be stripped.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

#: Env vars that leak developer/cluster state into otherwise-hermetic unit tests.
_LEAKABLE_ENVS = (
    "XRLENV_BENCHMARK_CACHE",
    "XRLENV_NODE_TOKEN",
    "XRLENV_CONSUMER_TOKEN",
    "XRLENV_OPERATOR_TOKEN",
    "XRLENV_GRPC_HOST",
    "XRLENV_GRPC_PORT",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip leakable ``XRLENV_*`` env from every unit test's starting state.

    Tests that want one set use ``monkeypatch.setenv`` (restored cleanly at
    teardown); nothing here survives a test boundary.
    """
    for env in _LEAKABLE_ENVS:
        monkeypatch.delenv(env, raising=False)
    yield
