"""Dashboard overview rows — in particular the solved-only latency/cost medians.

These are ADDITIONAL columns, never a replacement for the all-task ones: the two answer different
questions. "What does attempting a task cost" mixes in failures, which can be cheap (died early) or
expensive (burned the whole budget getting nowhere); "what does solving one cost" is the number to
compare across harnesses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "scripts"))
results_data = pytest.importorskip("results_data", reason="dashboard helper lives under experiments/")

ALL_LAT, ALL_COST = "Latency/task (s)", "Cost/task ($)"
SOLVED_LAT, SOLVED_COST = "[solved]Latency/task (s)", "[solved]Cost/task ($)"
_PRICES = {"*": {"input": 1.0, "cached": 0.0, "output": 10.0}}


def _tokens(prompt: int = 0, completion: int = 0) -> dict:
    return {"prompt": prompt, "completion": completion, "input_uncached": prompt,
            "cache_read": 0, "cache_write": 0, "total": prompt + completion}


def _trial(resolved: bool, seconds: float, prompt: int, completion: int = 0) -> dict:
    return {"resolved": resolved, "agent_seconds": seconds, "tokens": _tokens(prompt, completion)}


def _summary(trials: list[dict]) -> list[dict]:
    return [{
        "runname": "r", "harness": "h", "model": "m", "effort": "medium", "in_progress": False,
        "benchmarks": {"b": {"num_tasks": len(trials),
                             "num_resolved": sum(1 for t in trials if t["resolved"]),
                             "score": 0.5, "median_latency_sec": 100.0,
                             "tokens": _tokens(100, 10), "trials": trials}},
    }]


def _row(trials: list[dict]) -> dict:
    return results_data.overview_rows(_summary(trials), _PRICES)[0]


def test_solved_medians_ignore_failed_trials() -> None:
    """A failure that burned 10x the budget must not drag the solved-only median with it."""
    row = _row([
        _trial(True, 100.0, 1_000_000),
        _trial(True, 200.0, 2_000_000),
        _trial(False, 9000.0, 90_000_000),      # expensive failure
    ])
    assert row[SOLVED_LAT] == 150             # median(100, 200) — the failure is excluded
    assert row[SOLVED_COST] == pytest.approx(1.5, abs=0.01)


def test_all_task_columns_are_left_untouched() -> None:
    """The existing columns keep their all-attempted-tasks meaning — the new ones are additive."""
    trials = [_trial(True, 100.0, 1_000_000), _trial(False, 300.0, 3_000_000)]
    row = _row(trials)
    assert row[ALL_LAT] == 100                # unchanged: precomputed median_latency_sec
    assert row[ALL_COST] == pytest.approx(2.0, abs=0.01)   # median over BOTH trials
    assert row[SOLVED_LAT] == 100             # solved-only sees just the one resolved trial
    assert row[SOLVED_COST] == pytest.approx(1.0, abs=0.01)
    assert row[ALL_COST] != row[SOLVED_COST]  # the distinction is real, not cosmetic


def test_solved_columns_are_none_when_nothing_was_solved() -> None:
    """A benchmark that solved nothing shows blanks, not zeros — 0 would read as 'free'."""
    row = _row([_trial(False, 500.0, 5_000_000), _trial(False, 600.0, 6_000_000)])
    assert row[SOLVED_LAT] is None and row[SOLVED_COST] is None


def test_solved_columns_tolerate_missing_timing_and_tokens() -> None:
    """A resolved trial with no timing still counts for cost, and vice versa — neither
    partial-data case may raise or silently drop the whole column."""
    untimed = {"resolved": True, "tokens": _tokens(1_000_000)}          # no agent_seconds
    untokened = {"resolved": True, "agent_seconds": 50.0, "tokens": _tokens()}   # no tokens
    row = _row([untimed, untokened])
    assert row[SOLVED_LAT] == 50                                        # from the untokened trial
    assert row[SOLVED_COST] == pytest.approx(1.0, abs=0.01)             # from the untimed trial


def test_columns_are_always_emitted_so_the_ui_can_hide_them() -> None:
    """The toggle is presentational: the data layer always produces the keys, and the dashboard
    drops them when the viewer hasn't opted in. A missing key would KeyError on drop()."""
    row = _row([_trial(True, 10.0, 10)])
    assert SOLVED_LAT in row and SOLVED_COST in row


def _dashboard_src() -> str:
    return (Path(__file__).resolve().parents[2] / "experiments/scripts/dashboard.py").read_text()


def test_dashboard_drops_exactly_the_columns_it_declares() -> None:
    """The drop list in the dashboard must name the producer's keys verbatim — a rename on one
    side would either KeyError or silently leak the columns into the default view."""
    src = _dashboard_src()
    assert f'"{SOLVED_LAT}", "{SOLVED_COST}"' in src
    assert "show_solved" in src and "if show_solved else solved_cols" in src


def test_dashboard_drop_survives_a_missing_column() -> None:
    """A stale import must not take down the page. Streamlit hot-reloads dashboard.py but keeps
    an already-imported results_data, so an editing session can pair a new drop-list with an old
    row builder -- which raised KeyError and blanked the whole Overview."""
    assert 'errors="ignore"' in _dashboard_src()

    import pandas as pd
    df = pd.DataFrame([{"Benchmark": "b", "In progress": False}])       # solved cols absent
    out = df.drop(columns=["In progress", SOLVED_LAT, SOLVED_COST], errors="ignore")
    assert list(out.columns) == ["Benchmark"]


def test_displayed_label_matches_the_data_key() -> None:
    """The column_config label must equal the key, so the header a reader sees is the field the
    row actually holds -- they diverged once and the table showed the all-task names."""
    src = _dashboard_src()
    for key in (SOLVED_LAT, SOLVED_COST):
        assert f'"{key}": st.column_config.NumberColumn(\n                    "{key}"' in src


def test_toggle_explains_itself_when_the_columns_are_absent() -> None:
    """errors="ignore" keeps the page alive but makes a stale session SILENT -- the checkbox
    appears to do nothing. The absence has to be explained, with the fix (restart streamlit),
    or the next person debugs the toggle instead of their process."""
    src = _dashboard_src()
    assert "not any(c in df.columns for c in solved_cols)" in src
    assert "restart streamlit" in src.lower()


# --- harness version ---------------------------------------------------------


def test_version_column_sits_immediately_after_harness() -> None:
    """Asked for explicitly: the version must read next to the harness it qualifies, not be
    parked at the end of a wide table."""
    row = _row([_trial(True, 10.0, 10)])
    cols = list(row)
    assert cols[cols.index("Harness") + 1] == "Version"


def test_version_combines_the_declared_version_with_the_ref(monkeypatch, tmp_path) -> None:
    """`<version>_<ref6>`: the version alone cannot tell two runs apart once evolution starts
    (every candidate shares the baseline's version), and the ref alone loses the name the
    harness was onboarded under. Both halves earn their place."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("agent:\n  harness:\n    name: monet\n    version: '20260826'\n", encoding="utf-8")
    monkeypatch.setattr(results_data, "_MANIFEST_DIR", tmp_path / "none")
    results_data._versions_by_ref.cache_clear()
    got = results_data.harness_version({"source": {"ref": "f0d15a044922aaaa"}}, str(cfg))
    assert got == "20260826_f0d15a"


def test_two_candidates_of_one_version_are_distinguishable(monkeypatch, tmp_path) -> None:
    """The reason the ref is appended at all: an evolution run's candidates all declare the
    baseline version, so version-only labels would collapse them into one row identity."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("agent:\n  harness:\n    version: '20260826'\n", encoding="utf-8")
    monkeypatch.setattr(results_data, "_MANIFEST_DIR", tmp_path / "none")
    results_data._versions_by_ref.cache_clear()
    a = results_data.harness_version({"source": {"ref": "aaaaaa000000"}}, str(cfg))
    b = results_data.harness_version({"source": {"ref": "bbbbbb000000"}}, str(cfg))
    assert a != b and a.startswith("20260826_") and b.startswith("20260826_")


def test_version_falls_back_to_the_manifest_when_the_config_moved(monkeypatch, tmp_path) -> None:
    """Configs get reorganised; run.json keeps the path, not the value. The manifest still maps
    the pinned ref to the version it was onboarded under."""
    (tmp_path / "afcode_v9.json").write_text(
        '{"version": "v9.9.9", "ref": "abc123def456"}', encoding="utf-8")
    monkeypatch.setattr(results_data, "_MANIFEST_DIR", tmp_path)
    results_data._versions_by_ref.cache_clear()
    assert results_data.harness_version(
        {"source": {"ref": "abc123def456"}}, "/gone/config.yaml") == "v9.9.9_abc123"


def test_version_falls_back_to_the_ref_not_a_guess(monkeypatch, tmp_path) -> None:
    """Neither source available: the short ref is honest. Parsing one out of the run directory
    name would only be right for generator-produced names, and wrong silently otherwise."""
    monkeypatch.setattr(results_data, "_MANIFEST_DIR", tmp_path)   # empty
    results_data._versions_by_ref.cache_clear()
    assert results_data.harness_version({"source": {"ref": "f0d15a044922aaaa"}}) == "f0d15a044922"
    assert results_data.harness_version({}) == "?"


def test_unreadable_config_never_raises(monkeypatch, tmp_path) -> None:
    """A malformed or vanished config must degrade, not take down the whole dashboard."""
    monkeypatch.setattr(results_data, "_MANIFEST_DIR", tmp_path)
    results_data._versions_by_ref.cache_clear()
    bad = tmp_path / "bad.yaml"
    bad.write_text("agent: [unclosed\n", encoding="utf-8")
    assert results_data.harness_version({"source": {"ref": "abcdef123456"}}, str(bad)) == "abcdef123456"


def test_schema_bump_invalidates_cached_summaries() -> None:
    """`version` is a new summary field; without a bump, every cached summary.json would keep
    serving rows without it and the column would read '?' forever."""
    assert results_data._SCHEMA >= 4
