"""Regression guard for audit H9 — every ``run_full_sweep.sh`` must source ``.env`` BEFORE it
resolves the cache root, so ONE immutable ``XRLENV_BENCHMARK_CACHE`` drives every stage.

H9: when the wrapper resolved ``XRLENV_BENCHMARK_CACHE`` / ``SHARD`` BEFORE sourcing ``.env``,
a direct full-sweep run could gate the green set against one cache (the pre-source value) while
the Python build/eval children inherited a different one (the post-source ``.env`` value) — i.e.
gate one corpus and execute another. The fix makes ``.env`` (which carries both CP creds AND the
cache constant) load first, then resolves the root once. This test encodes that ordering.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_BENCH_DIR = Path(__file__).resolve().parents[2]  # xrlenv_plugins/benchmarks/tests/integration/ -> benchmarks
_WRAPPERS = [
    "swebench_verified", "deep_swe", "lhtb", "seta", "terminal_bench_2_1", "terminalworld",
]


def _line_of(lines: list[str], pattern: str) -> int:
    rx = re.compile(pattern)
    for i, ln in enumerate(lines):
        if rx.search(ln):
            return i
    raise AssertionError(f"pattern {pattern!r} not found")


@pytest.mark.parametrize("bench", _WRAPPERS)
def test_env_sourced_before_cache_resolution(bench: str) -> None:
    lines = (_BENCH_DIR / bench / "run_full_sweep.sh").read_text().splitlines()
    source = _line_of(lines, r"source \./\.env")
    resolve = _line_of(lines, r'^: "\$\{XRLENV_BENCHMARK_CACHE:=')
    shard = _line_of(lines, r"^SHARD=")
    # .env must load first, so the ONE resolved root below drives shell + Python stages (H9).
    assert source < resolve < shard, (
        f"{bench}: .env source (line {source + 1}) must precede cache resolution "
        f"(line {resolve + 1}) and SHARD (line {shard + 1}) — else the shell gate and the "
        f"Python children can resolve different cache roots (audit H9)."
    )


@pytest.mark.parametrize("bench", _WRAPPERS)
def test_wrapper_rejects_dest_cache_override(bench: str) -> None:
    # audit H9 (second input): a pass-through --dest would reach the evaluator (argparse
    # last-wins) and evaluate a DIFFERENT cache than the wrapper gated. The wrapper must reject
    # a cache-root override at its boundary, before any build/gate/run.
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    wrapper = _BENCH_DIR / bench / "run_full_sweep.sh"
    for override in (["--dest", "/tmp/other"], ["--dest=/tmp/other"], ["--cache", "/tmp/other"]):
        res = subprocess.run(
            ["bash", str(wrapper), *override, "--list-green", "--skip-build-cache"],
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                 "XRLENV_BENCHMARK_CACHE": "/tmp/x"},
            capture_output=True, text=True,
        )
        assert res.returncode != 0, f"{bench}: {override} was NOT rejected"
        assert "not accepted" in res.stderr


_DEST_EVALUATORS = {
    "deep_swe": "xrlenv_plugins.benchmarks.deep_swe.run_oracle_sweep",
    "lhtb": "xrlenv_plugins.benchmarks.lhtb.run_oracle_sweep",
    "terminal_bench_2_1": "xrlenv_plugins.benchmarks.terminal_bench_2_1.run_oracle_sweep",
    "terminalworld": "xrlenv_plugins.benchmarks.terminalworld.run_oracle_sweep",
}
# Every abbreviation / equals form of the cache-root flags. The wrapper blocks the EXACT
# `--dest`/`--cache` forms; the evaluator parser (allow_abbrev=False) must reject the PREFIXES
# so none of `--d`/`--de`/`--des`/`--d=`/`--ca` can smuggle a cache override past the gate (H9).
_ABBREV_FORMS = ["--d", "--de", "--des", "--d=/tmp/b", "--des=/tmp/b", "--ca", "--cache"]


@pytest.mark.parametrize("modpath", _DEST_EVALUATORS.values(), ids=_DEST_EVALUATORS.keys())
@pytest.mark.parametrize("form", _ABBREV_FORMS)
def test_evaluator_rejects_dest_abbreviations(modpath: str, form: str) -> None:
    # audit H9 (root cause): argparse allow_abbrev=True let --des/--de/--d resolve to --dest,
    # bypassing the wrapper's exact-form reject. allow_abbrev=False must reject every prefix so
    # the executed corpus can never differ from the gated one via an abbreviated cache override.
    import importlib

    m = importlib.import_module(modpath)
    argv = form.split() if "=" in form else [form, "/tmp/b"]
    with pytest.raises(SystemExit) as ei:
        m.main(argv)
    # argparse rejects the unknown/abbreviated flag with exit code 2 (never 0 / a silent accept).
    assert ei.value.code not in (0, None)


@pytest.mark.parametrize("bench", _WRAPPERS)
def test_list_green_skips_env_source(bench: str) -> None:
    # The pure --list-green read needs no CP creds and must NOT source .env (so a caller's
    # fixture cache wins) — the source stays guarded behind `if [ "$LIST_GREEN" != 1 ]`.
    text = (_BENCH_DIR / bench / "run_full_sweep.sh").read_text()
    assert 'if [ "$LIST_GREEN" != 1 ]; then' in text
    # the source line lives inside that guard (next non-blank line after it).
    lines = text.splitlines()
    guard = _line_of(lines, r'if \[ "\$LIST_GREEN" != 1 \]; then')
    assert "source ./.env" in lines[guard + 1]
