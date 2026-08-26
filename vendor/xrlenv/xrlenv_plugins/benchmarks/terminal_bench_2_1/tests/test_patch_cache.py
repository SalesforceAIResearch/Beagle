"""Unit tests for the pure patch logic in ``build_cache.apply_solve_patch``.

The populate stages (seed copytree / registry pull) are I/O + network and
covered operationally; the load-bearing, deterministic piece is the
one-line-pin insertion, which these tests pin down.
"""

from __future__ import annotations

import re
import tomllib

import pytest
from xrlenv_plugins.benchmarks.terminal_bench_2_1.build_cache import (
    ENV_PATCHES,
    PATCHES,
    SolvePatch,
    TaskEnvPatch,
    apply_solve_patch,
    apply_task_env_patch,
)

# The real build-cython-ext pin, referenced by several tests.
CYTHON = next(p for p in PATCHES if p.task == "build-cython-ext")

_UNPINNED = (
    "#!/bin/bash\n"
    "git clone --depth 1 --branch 0.5.3 https://x/pyknotid.git /app/pyknotid\n"
    "pip install setuptools==80.9.0 cython==3.1.3\n"
    "cd /app/pyknotid || exit\n"
    "python setup.py build_ext --inplace\n"
    "pip install -e .\n"
)


def test_inserts_pin_immediately_before_editable_install() -> None:
    new_text, status = apply_solve_patch(_UNPINNED, CYTHON)
    assert status == "patched"
    lines = new_text.splitlines()
    idx = lines.index("pip install -e .")
    assert lines[idx - 1] == "pip install 'planarity==0.6'"
    # Nothing else moved: the tail is exactly pin + editable install.
    assert lines[-2:] == ["pip install 'planarity==0.6'", "pip install -e ."]


def test_patched_text_is_stable_under_reapplication() -> None:
    once, _ = apply_solve_patch(_UNPINNED, CYTHON)
    twice, status = apply_solve_patch(once, CYTHON)
    assert status == "already"
    assert twice == once  # idempotent — no double insertion


def test_sentinel_short_circuits_even_without_anchor() -> None:
    # Already-pinned content with the sentinel but no bare ``-e .`` anchor
    # must be reported ``already`` rather than raising for a missing anchor.
    text = "pip install 'planarity==0.6'\npip install -e . --no-deps\n"
    out, status = apply_solve_patch(text, CYTHON)
    assert status == "already"
    assert out == text


def test_missing_anchor_fails_loud() -> None:
    text = "#!/bin/bash\npython setup.py build_ext --inplace\n"
    with pytest.raises(ValueError, match=r"anchor .* not found"):
        apply_solve_patch(text, CYTHON)


def test_crlf_line_endings_preserved_on_inserted_line() -> None:
    patch = SolvePatch(
        task="x",
        reason="test",
        anchor=re.compile(r"^\s*pip install -e \.\s*$"),
        insert_line="pip install 'planarity==0.6'",
        sentinel=re.compile(r"planarity==0\.6"),
    )
    text = "cd /app\r\npip install -e .\r\n"
    out, status = apply_solve_patch(text, patch)
    assert status == "patched"
    assert "pip install 'planarity==0.6'\r\n" in out


def test_indented_anchor_matches() -> None:
    patch = SolvePatch(
        task="x",
        reason="test",
        anchor=re.compile(r"^\s*pip install -e \.\s*$"),
        insert_line="PIN",
        sentinel=re.compile(r"^PIN$", re.MULTILINE),
    )
    text = "if true; then\n    pip install -e .\nfi\n"
    out, status = apply_solve_patch(text, patch)
    assert status == "patched"
    assert "PIN\n    pip install -e .\n" in out


# ── cpuset-pinning marker (task.toml [environment.env]) ───────────────────────

# A minimal task.toml with the (usually empty) [environment.env] table the
# marker anchors on, plus a following table to catch mis-nesting.
_TASK_TOML = (
    'schema_version = "1.1"\n\n'
    "[environment]\n"
    "cpus = 2\n"
    "memory_mb = 4096\n\n"
    "[environment.env]\n\n"
    "[solution.env]\n"
)

# The real install-windows-3.11 marker, referenced by several tests.
IWIN = next(p for p in ENV_PATCHES if p.task == "install-windows-3.11")


def test_env_marker_lands_under_environment_env_table() -> None:
    new_text, status = apply_task_env_patch(_TASK_TOML, IWIN)
    assert status == "patched"
    doc = tomllib.loads(new_text)
    # The marker must belong to [environment.env] — not leak into the
    # following [solution.env] table (the classic TOML mis-nesting trap).
    assert doc["environment"]["env"] == {"XRLENV_CPU_PINNING": "1"}
    assert doc["solution"]["env"] == {}
    # Declared limits untouched.
    assert doc["environment"]["cpus"] == 2
    assert doc["environment"]["memory_mb"] == 4096


def test_env_marker_is_idempotent() -> None:
    once, _ = apply_task_env_patch(_TASK_TOML, IWIN)
    twice, status = apply_task_env_patch(once, IWIN)
    assert status == "already"
    assert twice == once  # no double insertion


def test_env_marker_missing_table_fails_loud() -> None:
    text = '[task]\nname = "x"\n'
    with pytest.raises(ValueError, match=r"\[environment\.env\] table not found"):
        apply_task_env_patch(text, IWIN)


def test_env_marker_crlf_preserved() -> None:
    patch = TaskEnvPatch(
        task="x",
        reason="test",
        env_key="XRLENV_CPU_PINNING",
        env_value="1",
        sentinel=re.compile(r"XRLENV_CPU_PINNING\s*="),
    )
    text = "[environment.env]\r\n[solution.env]\r\n"
    out, status = apply_task_env_patch(text, patch)
    assert status == "patched"
    assert 'XRLENV_CPU_PINNING = "1"\r\n' in out


def test_env_patches_cover_expected_nproc_scaling_tasks() -> None:
    tasks = {p.task for p in ENV_PATCHES}
    assert tasks == {
        "install-windows-3.11",
        "caffe-cifar-10",
        "build-pov-ray",
        "rstan-to-pystan",
        "sqlite-with-gcov",
        # Not an nproc-scaled *build* like the five above: torch sizes its
        # thread pool from sched_getaffinity and throttles against the declared
        # cpus=1 CFS quota (676.0s unpinned vs 59.6s pinned, uncontended).
        "torch-pipeline-parallelism",
    }
    # Every marker sets the same truthy env key the plugin reads.
    for p in ENV_PATCHES:
        assert p.env_key == "XRLENV_CPU_PINNING"
        assert p.env_value == "1"
        assert p.reason  # each carries an audit rationale
