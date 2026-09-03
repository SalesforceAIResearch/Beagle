"""One cache-root contract across every benchmark kit.

``XRLENV_BENCHMARK_CACHE`` (or an explicit ``--dest``) is the ONLY way a kit learns where
the task cache lives. ``_benchmark_cache.benchmark_cache_root`` is the single implementation:
it rejects the retired var/path and raises when nothing is set.

Every ``build_plan_gen.py`` used to re-implement that lookup with a
``~/.cache/harbor/tasks`` fallback, so with the var unset a generator resolved to a
directory that need not exist, discovered 0 tasks, and emitted an **empty warm plan with no
error** — a plan that warms nothing. ``build_cache.py`` raised on the very same input, so a
single variable had two contracts inside one kit.

These tests are shared rather than copied per kit so a NEW kit is covered the day it lands.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1]
# Paths a shipped file may never bake: a home dir, or this fleet's shared mount.
BAKED_PREFIXES = ("~/", "/fsx/", "/home/")


# The ONE reviewed exception. terminal_bench_2_1's DEFAULT_SEED_DIR is harbor's own
# export location (`harbor tasks export` writes ~/.cache/harbor/tasks/<dataset>), used as
# the default of the optional `--seed-dir` flag and printed in `--help`. It is a third-
# party tool's documented output path, not our cache root, and a wrong value fails
# visibly (the seed dir simply is not there) rather than silently resolving to an empty
# one. Everything else must come from XRLENV_BENCHMARK_CACHE.
ALLOWED = {"terminal_bench_2_1/build_cache.py"}


def _kits_with_shard_dir() -> list[str]:
    """Kits whose plan generator exposes ``_shard_dir`` — the shape this contract covers.

    ``seta`` and ``swebench_verified`` resolve the root differently (a ``--cache-root``
    flag defaulting to the var, and a manifest fallback for the docker-py drop-in), and
    neither ever fell back to a home directory, so they are correctly out of scope here.
    """
    kits = []
    for path in sorted(BENCHMARKS.glob("*/build_plan_gen.py")):
        if "_shard_dir" in path.read_text(encoding="utf-8"):
            kits.append(path.parent.name)
    return kits


def test_the_kit_list_is_not_silently_empty() -> None:
    """A glob that matched nothing would make every test below vacuously pass."""
    kits = _kits_with_shard_dir()
    assert len(kits) >= 6, kits


@pytest.mark.parametrize("kit", _kits_with_shard_dir())
def test_shard_dir_fails_loud_when_the_cache_root_is_unset(
    kit: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REGRESSION. No kit may fall back to a default location: an unset root is an
    operator error, and answering it with a plausible-but-wrong directory produces a
    silently empty plan instead of a message."""
    module = importlib.import_module(f"xrlenv_plugins.benchmarks.{kit}.build_plan_gen")
    monkeypatch.delenv("XRLENV_BENCHMARK_CACHE", raising=False)
    with pytest.raises(SystemExit, match="XRLENV_BENCHMARK_CACHE"):
        module._shard_dir()


@pytest.mark.parametrize("kit", _kits_with_shard_dir())
def test_shard_dir_follows_the_cache_root(
    kit: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(f"xrlenv_plugins.benchmarks.{kit}.build_plan_gen")
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(tmp_path))
    assert module._shard_dir() == tmp_path / module.SHARD


def _baked_literals(path: Path) -> list[str]:
    """String LITERALS under ``path`` that start with a baked prefix, excluding docstrings.

    Checked over the AST rather than raw text so prose naming a path — a docstring
    explaining why a fallback was removed, a README example in a module header — is not a
    false positive, while an actual ``Path("~/.cache/...")`` is.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        ast.get_docstring(n, clean=False)
        for n in ast.walk(tree)
        if isinstance(n, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    try:
        name = path.relative_to(BENCHMARKS).as_posix()
    except ValueError:                       # a scratch file from the self-test below
        name = path.name
    return [
        f"{name}:{n.lineno}: {n.value!r}"
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value not in docstrings
        and n.value.startswith(BAKED_PREFIXES)
    ]


def test_no_kit_bakes_a_home_or_site_directory() -> None:
    """The positive statement of the same rule, over every shipped kit module."""
    offenders: list[str] = []
    for path in sorted(BENCHMARKS.rglob("*.py")):
        if "__pycache__" in path.parts or path.name.startswith("test_"):
            continue
        if path.relative_to(BENCHMARKS).as_posix() in ALLOWED:
            continue
        offenders += _baked_literals(path)
    assert not offenders, offenders


def test_the_guard_detects_a_reintroduced_fallback(tmp_path: Path) -> None:
    """The guard above only means something if it FAILS on the code it forbids — a
    docstring-only exclusion bug would make it silently permissive."""
    good = tmp_path / "good.py"
    good.write_text('"""Explains ~/.cache/harbor/tasks in prose."""\nx = 1\n')
    assert _baked_literals(good) == []

    bad = tmp_path / "bad.py"
    bad.write_text('from pathlib import Path\nd = Path("~/.cache/harbor/tasks")\n')
    assert len(_baked_literals(bad)) == 1
