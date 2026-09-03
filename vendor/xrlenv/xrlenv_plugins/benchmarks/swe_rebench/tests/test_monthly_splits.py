"""Checks for the monthly-split index + its fetcher (no network, no cluster).

``scripts/monthly_splits.json`` is written by
``scripts/fetch_monthly_splits.py`` from the HF datasets-server, because the
Harbor Hub package this kit downloads is the flat ``test`` split and records no
month. These tests pin what the ``--split`` selector in ``run_full_sweep.sh``
relies on, and cover the fetcher's pure logic against fake payloads.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from xrlenv_plugins.benchmarks.swe_rebench.scripts import fetch_monthly_splits as fms

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
INDEX = SCRIPTS / "monthly_splits.json"

# Pinned from upstream (nebius/SWE-rebench-leaderboard, 2026-09-01): 15 monthly
# splits whose union is the 860-task `test` split. Upstream adds a split monthly,
# so a failure here means "regenerate the index":
#   .venv/bin/python -m ….scripts.fetch_monthly_splits
EXPECTED_TASKS = 860
EXPECTED_COUNTS = {
    "2025_01": 109, "2025_02": 76, "2025_03": 62, "2025_04": 40, "2025_05": 40,
    "2025_06": 40, "2025_07": 30, "2025_08": 52, "2025_09": 50, "2025_10": 51,
    "2025_11": 47, "2025_12": 48, "2026_01": 48, "2026_02": 57, "2026_03": 110,
}


@pytest.fixture(scope="module")
def index() -> dict[str, Any]:
    assert INDEX.is_file(), f"missing index: {INDEX}"
    return json.loads(INDEX.read_text())


# ── the committed index ───────────────────────────────────────────────────────


def test_index_shape(index: dict[str, Any]) -> None:
    assert set(index) == {"dataset", "task_count", "splits"}
    assert index["dataset"] == fms.DEFAULT_DATASET
    assert isinstance(index["splits"], dict)


def test_per_split_counts_match_upstream(index: dict[str, Any]) -> None:
    counts = {name: len(ids) for name, ids in index["splits"].items()}
    assert counts == EXPECTED_COUNTS


def test_task_count_agrees_with_the_splits(index: dict[str, Any]) -> None:
    every = [t for ids in index["splits"].values() for t in ids]
    assert index["task_count"] == EXPECTED_TASKS == len(every)


def test_splits_are_disjoint(index: dict[str, Any]) -> None:
    """`--split a,b` unions the members; an overlap would silently double-count."""
    every = [t for ids in index["splits"].values() for t in ids]
    assert len(set(every)) == len(every)


def test_split_names_are_well_formed(index: dict[str, Any]) -> None:
    for name in index["splits"]:
        assert re.fullmatch(r"20\d{2}_(0[1-9]|1[0-2])", name), name


def test_ids_are_sorted_within_each_split(index: dict[str, Any]) -> None:
    """Deterministic output, so a no-op regeneration is a zero-line diff."""
    for name, ids in index["splits"].items():
        assert ids == sorted(ids), f"{name} is unsorted"


def test_committed_file_is_exactly_what_render_produces(index: dict[str, Any]) -> None:
    """Guards against a hand-edit: the file must be byte-identical to the
    script's own rendering of its content."""
    assert INDEX.read_text() == fms.render(index)


def test_index_covers_the_smoke_set() -> None:
    smoke = SCRIPTS / "smoke_30tasks.txt"
    if not smoke.is_file():
        pytest.skip("no smoke manifest")
    known = {
        t for ids in json.loads(INDEX.read_text())["splits"].values() for t in ids
    }
    smoke_ids = [
        line.split()[0]
        for line in smoke.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert smoke_ids
    assert set(smoke_ids) <= known, sorted(set(smoke_ids) - known)


# ── the fetcher's pure logic (fake payloads; no network) ──────────────────────


def _fake_server(monkeypatch: pytest.MonkeyPatch, splits: dict[str, list[str]]) -> None:
    """Stub ``_get`` with a datasets-server-shaped response, honouring paging."""
    def fake(url: str, *, timeout: float = 90.0) -> Any:
        if "/splits?" in url:
            names = [*splits, fms._UNION_SPLIT]
            return {"splits": [{"split": s} for s in names]}
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(url).query)
        split, offset, length = q["split"][0], int(q["offset"][0]), int(q["length"][0])
        ids = splits[split]
        page = ids[offset:offset + length]
        return {
            "rows": [{"row": {"instance_id": i}} for i in page],
            "num_rows_total": len(ids),
        }
    monkeypatch.setattr(fms, "_get", fake)


def test_build_index_groups_and_sorts(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_server(monkeypatch, {"2025_02": ["b", "a"], "2025_01": ["c"]})
    built = fms.build_index("some/dataset")
    assert built["task_count"] == 3
    assert built["splits"] == {"2025_01": ["c"], "2025_02": ["a", "b"]}
    assert built["dataset"] == "some/dataset"


def test_build_index_excludes_the_union_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """`test` is the union of the months — including it would double every id."""
    _fake_server(monkeypatch, {"2025_01": ["a", "b"]})
    assert set(fms.build_index("d")["splits"]) == {"2025_01"}


def test_fetch_split_ids_pages_past_the_100_row_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [f"task-{i:03d}" for i in range(250)]
    _fake_server(monkeypatch, {"2025_01": ids})
    assert fms.fetch_split_ids("d", "2025_01") == sorted(ids)


def test_build_index_rejects_overlapping_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_server(monkeypatch, {"2025_01": ["dup"], "2025_02": ["dup"]})
    with pytest.raises(SystemExit, match="appears in both"):
        fms.build_index("d")


def test_list_splits_fails_loud_when_only_the_union_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never write an empty index — that would make --split silently useless."""
    monkeypatch.setattr(
        fms, "_get", lambda url, timeout=90.0: {"splits": [{"split": "test"}]},
    )
    with pytest.raises(SystemExit, match="no split besides"):
        fms.list_splits("d")


def test_fetch_split_ids_fails_loud_on_a_short_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server ending the walk early (an empty page before `num_rows_total`)
    must not silently commit a partial split."""
    calls = {"n": 0}

    def fake(url: str, *, timeout: float = 90.0) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"rows": [{"row": {"instance_id": "a"}}], "num_rows_total": 99}
        return {"rows": [], "num_rows_total": 99}      # truncated walk

    monkeypatch.setattr(fms, "_get", fake)
    with pytest.raises(SystemExit, match=r"1 row\(s\), expected 99"):
        fms.fetch_split_ids("d", "2025_01")


def test_fetch_split_ids_fails_loud_on_a_schema_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fms, "_get",
        lambda url, timeout=90.0: {"rows": [{"row": {"id": "a"}}], "num_rows_total": 1},
    )
    with pytest.raises(SystemExit, match="no 'instance_id'"):
        fms.fetch_split_ids("d", "2025_01")


def test_render_is_deterministic() -> None:
    a = {"dataset": "d", "task_count": 1, "splits": {"2025_01": ["x"]}}
    b = {"splits": {"2025_01": ["x"]}, "task_count": 1, "dataset": "d"}
    assert fms.render(a) == fms.render(b)
    assert fms.render(a).endswith("\n")
