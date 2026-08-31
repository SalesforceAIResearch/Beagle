"""Shared fixtures for the beagle test suite.

Tier layout (mirrors the vendored xrlenv suite):

- ``tests/unit/``        — mechanical, hermetic, repeated-run. No cluster, no
  Docker, no money. This is what ``pytest -q`` runs by default.
- ``tests/smoke/``       — real end-to-end runs against a live xrlenv cluster
  (pull task images, drive harbor, cost minutes/money). **Excluded** from the
  default run via ``--ignore=tests/smoke``; opt in explicitly.
- ``tests/integration/`` — CI/CD orchestration (``run_all_smoke.sh`` runs the
  smokes back-to-back and fails loudly on any regression).

Env isolation for the unit tier lives in ``tests/unit/conftest.py`` (it strips
the leakable ``XRLENV_*`` env so unit tests are hermetic). Smoke tests
deliberately keep the operator's environment — they need the live cluster.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def package_root(repo_root: Path) -> Path:
    return repo_root / "beagle"
