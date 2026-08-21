"""Unit tests for the swebench-verified cache builder's pure/filesystem logic.

Network-free: the HuggingFace ``populate`` fetch is not exercised here. These tests
cover ``_row_to_cache`` (pure validation + file-set derivation) and the idempotent
``_materialize`` function.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from xrlenv_plugins.benchmarks.swebench_verified.build_cache import (
    ANCHOR,
    _is_complete,
    _materialize,
    _row_to_cache,
)

# ── _row_to_cache ─────────────────────────────────────────────────────────────


def _valid_row(**overrides: object) -> dict[str, object]:
    """Minimal valid upstream row with required fields."""
    base = {
        "instance_id": "astropy__astropy-7166",
        "patch": "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n",
        "problem_statement": "Fix the broken thing in astropy.",
        "repo": "astropy/astropy",
        "version": "7166",
    }
    base.update(overrides)
    return base


def test_row_to_cache_returns_three_files() -> None:
    files = _row_to_cache(_valid_row())
    assert set(files) == {"instance.json", "problem_statement.md", "gold_patch.diff"}


def test_row_to_cache_instance_json_is_full_row_serialized() -> None:
    row = _valid_row(extra_field="extra_value")
    files = _row_to_cache(row)
    parsed = json.loads(files["instance.json"])
    assert parsed["instance_id"] == "astropy__astropy-7166"
    assert parsed["extra_field"] == "extra_value"


def test_row_to_cache_problem_statement_md_matches_row_field() -> None:
    row = _valid_row(problem_statement="Describe the issue clearly.")
    files = _row_to_cache(row)
    assert files["problem_statement.md"] == "Describe the issue clearly."


def test_row_to_cache_gold_patch_diff_matches_row_patch() -> None:
    patch_text = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+fix\n"
    row = _valid_row(patch=patch_text)
    files = _row_to_cache(row)
    assert files["gold_patch.diff"] == patch_text


@pytest.mark.parametrize(
    "missing_field",
    ["instance_id", "patch", "problem_statement"],
)
def test_row_to_cache_raises_on_missing_required_field(missing_field: str) -> None:
    row = _valid_row()
    del row[missing_field]
    with pytest.raises(SystemExit, match="missing"):
        _row_to_cache(row)


@pytest.mark.parametrize(
    "empty_field",
    ["instance_id", "patch", "problem_statement"],
)
def test_row_to_cache_raises_on_empty_required_field(empty_field: str) -> None:
    row = _valid_row(**{empty_field: ""})
    with pytest.raises(SystemExit, match="missing"):
        _row_to_cache(row)


def test_row_to_cache_error_message_mentions_field_names() -> None:
    row = _valid_row()
    del row["patch"]
    del row["problem_statement"]
    with pytest.raises(SystemExit) as exc_info:
        _row_to_cache(row)
    msg = str(exc_info.value)
    assert "patch" in msg or "problem_statement" in msg


def test_row_to_cache_error_message_includes_instance_id_when_present() -> None:
    row = _valid_row(patch="")
    with pytest.raises(SystemExit) as exc_info:
        _row_to_cache(row)
    assert "astropy__astropy-7166" in str(exc_info.value)


# ── _materialize (idempotent) ─────────────────────────────────────────────────


def test_materialize_writes_files_and_returns_true_on_first_call(tmp_path: Path) -> None:
    shard = tmp_path / "swebench-verified"
    row = _valid_row()
    result = _materialize(row, shard)
    assert result is True

    inst_dir = shard / "astropy__astropy-7166"
    assert (inst_dir / "instance.json").is_file()
    assert (inst_dir / "problem_statement.md").is_file()
    assert (inst_dir / "gold_patch.diff").is_file()


def test_materialize_instance_json_content_is_valid_json(tmp_path: Path) -> None:
    shard = tmp_path / "swebench-verified"
    row = _valid_row()
    _materialize(row, shard)
    raw = (shard / "astropy__astropy-7166" / "instance.json").read_text()
    parsed = json.loads(raw)
    assert parsed["instance_id"] == "astropy__astropy-7166"


def test_materialize_is_idempotent_returns_false_on_second_call(tmp_path: Path) -> None:
    shard = tmp_path / "swebench-verified"
    row = _valid_row()
    first = _materialize(row, shard)
    assert first is True
    second = _materialize(row, shard)
    assert second is False


def test_materialize_repairs_corrupt_extract(tmp_path: Path) -> None:
    # audit M7: a sibling extract that no longer MATCHES the anchor is not complete — it
    # must be re-materialized (repaired), not skipped on the anchor's mere presence.
    shard = tmp_path / "swebench-verified"
    row = _valid_row(problem_statement="original statement")
    _materialize(row, shard)

    inst_dir = shard / "astropy__astropy-7166"
    (inst_dir / "problem_statement.md").write_text("CORRUPTED")  # no longer matches the anchor

    result = _materialize(_valid_row(problem_statement="original statement"), shard)
    assert result is True                                        # re-materialized (repaired)
    assert (inst_dir / "problem_statement.md").read_text() == "original statement"


def test_materialize_and_is_complete_roundtrip_crlf_problem_statement(tmp_path: Path) -> None:
    # REGRESSION (cache-completeness CRLF false-negative): a problem_statement / patch with
    # Windows CRLF line endings (common in GitHub issue text — 252 of the 500 Verified rows)
    # must materialize AND then pass _is_complete. Previously _is_complete read the extract with
    # read_text() (universal-newline: \r\n -> \n) but compared against the raw \r\n string, so
    # every CRLF-bearing instance was wrongly seen as incomplete → the H4 corpus gate reported
    # "252 missing" against a fully-built 500-instance cache and refused to run.
    shard = tmp_path / "swebench-verified"
    crlf_statement = "Line one.\r\nLine two.\r\nLine three.\r\n"
    crlf_patch = "diff --git a/f b/f\r\n--- a/f\r\n+++ b/f\r\n@@ -1 +1 @@\r\n+x\r\n"
    row = _valid_row(problem_statement=crlf_statement, patch=crlf_patch)

    assert _materialize(row, shard) is True
    inst_dir = shard / "astropy__astropy-7166"
    # the stored bytes are the VERBATIM dataset value (CRLF preserved) …
    assert (inst_dir / "problem_statement.md").read_bytes() == crlf_statement.encode("utf-8")
    assert (inst_dir / "gold_patch.diff").read_bytes() == crlf_patch.encode("utf-8")
    # … and the completeness check accepts it (was False before the byte-exact fix) …
    assert _is_complete(inst_dir) is True
    # … so a re-materialize is a no-op skip, not an endless re-write.
    assert _materialize(dict(row), shard) is False


def test_materialize_creates_instance_dir_if_absent(tmp_path: Path) -> None:
    shard = tmp_path / "swebench-verified"
    assert not shard.exists()
    _materialize(_valid_row(), shard)
    assert (shard / "astropy__astropy-7166").is_dir()


def test_materialize_anchor_is_instance_json(tmp_path: Path) -> None:
    shard = tmp_path / "swebench-verified"
    _materialize(_valid_row(), shard)
    assert (shard / "astropy__astropy-7166" / ANCHOR).is_file()


def test_materialize_rewrites_incomplete_dir_missing_siblings(tmp_path: Path) -> None:
    # audit M7: an anchor with MISSING siblings is INCOMPLETE — re-materialize, don't skip
    # on the anchor's mere presence.
    shard = tmp_path / "swebench-verified"
    inst_dir = shard / "astropy__astropy-7166"
    inst_dir.mkdir(parents=True)
    (inst_dir / ANCHOR).write_text("{}")  # anchor present, siblings absent -> incomplete

    result = _materialize(_valid_row(), shard)
    assert result is True                                    # rewritten, not skipped
    assert (inst_dir / "problem_statement.md").is_file()     # now complete
    assert (inst_dir / "gold_patch.diff").is_file()


def test_materialize_rewrites_corrupt_anchor(tmp_path: Path) -> None:
    # audit M7: a present-but-CORRUPT anchor is not complete — re-materialize it.
    shard = tmp_path / "swebench-verified"
    inst_dir = shard / "astropy__astropy-7166"
    inst_dir.mkdir(parents=True)
    (inst_dir / "instance.json").write_text("{ this is not valid json")
    (inst_dir / "problem_statement.md").write_text("x")
    (inst_dir / "gold_patch.diff").write_text("x")

    result = _materialize(_valid_row(), shard)
    assert result is True
    json.loads((inst_dir / ANCHOR).read_text())   # anchor is valid JSON now (no raise)


# ── atomic materialization (audit M7) ─────────────────────────────────────────


def test_materialize_leaves_no_temp_dir(tmp_path: Path) -> None:
    # the temp/stale swap siblings must not leak; a persistent .lock file IS expected (M7).
    shard = tmp_path / "swebench-verified"
    _materialize(_valid_row(), shard)
    swap_leftovers = [p.name for p in shard.iterdir()
                      if ".tmp-" in p.name or p.name.endswith(".old")]
    assert swap_leftovers == []
    real = [p.name for p in shard.iterdir() if not p.name.startswith(".")]
    assert real == ["astropy__astropy-7166"]


def test_materialize_replaces_anchorless_leftover(tmp_path: Path) -> None:
    # a prior crash left an anchor-LESS partial dir (the whole point of M7: an interrupted
    # write must never carry the anchor). A re-populate must treat it as incomplete and
    # replace it with a complete, atomically-swapped dir — not skip it, not merge into it.
    shard = tmp_path / "swebench-verified"
    inst_dir = shard / "astropy__astropy-7166"
    inst_dir.mkdir(parents=True)
    (inst_dir / "gold_patch.diff").write_text("STALE partial write")  # no anchor

    result = _materialize(_valid_row(), shard)
    assert result is True                                    # treated as incomplete -> rewritten
    assert (inst_dir / ANCHOR).is_file()                     # now complete
    assert (inst_dir / "problem_statement.md").is_file()
    assert (inst_dir / "gold_patch.diff").read_text() != "STALE partial write"


# ── M7 residual: semantic completeness + concurrent-writer safety ─────────────


def _write_full(inst_dir: Path, anchor_obj: object) -> None:
    inst_dir.mkdir(parents=True, exist_ok=True)
    (inst_dir / "problem_statement.md").write_text("s")
    (inst_dir / "gold_patch.diff").write_text("p")
    (inst_dir / ANCHOR).write_text(json.dumps(anchor_obj))


def test_is_complete_rejects_empty_or_mismatched_anchor(tmp_path: Path) -> None:
    d = tmp_path / "swebench-verified" / "astropy__astropy-7166"
    _write_full(d, {})                                   # bare {} -> missing required fields
    assert _is_complete(d) is False
    _write_full(d, {"instance_id": "WRONG", "patch": "p", "problem_statement": "s"})
    assert _is_complete(d) is False                      # id disagrees with the dir name
    _write_full(d, {"instance_id": "astropy__astropy-7166", "patch": "p",
                    "problem_statement": "s"})
    assert _is_complete(d) is True                       # complete + semantically valid


def test_materialize_concurrent_writers_do_not_crash(tmp_path: Path) -> None:
    # audit M7: without per-instance locking two writers race the move-aside/swap-in and one
    # hits FileNotFoundError. The flock serializes them; the dir ends complete.
    import threading

    shard = tmp_path / "swebench-verified"
    row = _valid_row()
    errors: list[Exception] = []

    def worker() -> None:
        try:
            _materialize(row, shard)
        except Exception as exc:  # the test's whole point is "no exception"
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert _is_complete(shard / "astropy__astropy-7166")


# ── H6: builder must not treat instance_id as a filesystem path ───────────────


def test_materialize_rejects_traversal_instance_id(tmp_path: Path) -> None:
    # audit H6: a dataset row's instance_id feeds the destination/lock/temp/rmtree paths.
    # A "../victim" id must be REJECTED before any fs op — not move/replace/delete a sibling.
    shard = tmp_path / "cache" / "swebench-verified"
    shard.mkdir(parents=True)
    victim = tmp_path / "cache" / "victim"          # a sibling OUTSIDE the shard
    victim.mkdir()
    (victim / "precious.txt").write_text("do not touch")

    row = _valid_row(instance_id="../victim")
    with pytest.raises(SystemExit, match="unsafe instance id"):
        _materialize(row, shard)

    assert (victim / "precious.txt").read_text() == "do not touch"   # untouched
    assert victim.is_dir()


def test_safe_instance_id_accepts_bare_rejects_escapes() -> None:
    from xrlenv_plugins.benchmarks.swebench_verified.build_cache import _safe_instance_id
    assert _safe_instance_id("astropy__astropy-7166") == "astropy__astropy-7166"
    for bad in ("../victim", "a/b", "..", ".", "/abs", ""):
        with pytest.raises(SystemExit, match="unsafe instance id"):
            _safe_instance_id(bad)


def test_is_complete_rejects_symlinked_instance_dir(tmp_path: Path) -> None:
    # audit Low: a symlinked instance dir must not be trusted (following it reads out of shard).
    shard = tmp_path / "swebench-verified"
    shard.mkdir(parents=True)
    real = tmp_path / "real-outside"
    real.mkdir()
    for f in ("instance.json", "problem_statement.md", "gold_patch.diff"):
        (real / f).write_text("x")
    link = shard / "astropy__astropy-7166"
    link.symlink_to(real)
    assert _is_complete(link) is False


def test_is_complete_rejects_symlinked_child_file(tmp_path: Path) -> None:
    # audit Low: a required child that is itself a SYMLINK (even inside a real dir) must not be
    # trusted — following it reads out of shard. Point gold_patch.diff at an out-of-shard file
    # whose content would otherwise match, and assert it's still rejected.
    shard = tmp_path / "swebench-verified"
    inst = shard / "astropy__astropy-7166"
    inst.mkdir(parents=True)
    row = _valid_row()
    files = _row_to_cache(row)
    (inst / "instance.json").write_text(files["instance.json"])
    (inst / "problem_statement.md").write_text(files["problem_statement.md"])
    outside = tmp_path / "outside-patch.diff"
    outside.write_text(files["gold_patch.diff"])          # matching content, but out of shard
    (inst / "gold_patch.diff").symlink_to(outside)        # child is a symlink
    assert _is_complete(inst) is False


def test_list_complete_returns_only_complete_dirs(tmp_path: Path) -> None:
    # audit M13: list_complete enumerates only semantically-complete dirs — a bare {} anchor
    # and a temp sibling are excluded.
    from xrlenv_plugins.benchmarks.swebench_verified.build_cache import list_complete
    shard = tmp_path / "swebench-verified"
    _materialize(_valid_row(instance_id="astropy__astropy-7166"), shard)  # complete
    bare = shard / "django__django-11099"
    bare.mkdir(parents=True)
    (bare / "instance.json").write_text("{}")             # incomplete (bare anchor)
    (shard / ".stale.tmp-abcd").mkdir()                   # builder temp sibling — skipped
    assert list_complete(str(tmp_path)) == ["astropy__astropy-7166"]


def test_list_complete_empty_when_shard_absent(tmp_path: Path) -> None:
    from xrlenv_plugins.benchmarks.swebench_verified.build_cache import list_complete
    assert list_complete(str(tmp_path)) == []


def test_reclaim_orphan_temps_matches_exact_mkdtemp_grammar(tmp_path: Path) -> None:
    # audit Low: only the EXACT ``.<id>.tmp-<8 chars>`` mkdtemp grammar is reclaimed — a
    # legitimately-named ``.legitimate.tmp-data`` dir and a real instance dir are NOT.
    from xrlenv_plugins.benchmarks.swebench_verified.build_cache import _reclaim_orphan_temps
    shard = tmp_path / "swebench-verified"
    shard.mkdir(parents=True)
    (shard / ".astropy__astropy-7166.tmp-a1b2c3d4").mkdir()   # real orphan (8-char suffix)
    (shard / ".django__django-11099.tmp-zz00__ab").mkdir()    # real orphan (8-char suffix)
    (shard / ".legitimate.tmp-data").mkdir()                  # 4-char suffix — NOT an orphan
    (shard / "astropy__astropy-7166").mkdir()                 # a real instance dir — untouched
    (shard / ".astropy__astropy-7166.lock").write_text("")    # a lock file — untouched
    removed = sorted(_reclaim_orphan_temps(shard))
    assert removed == [".astropy__astropy-7166.tmp-a1b2c3d4", ".django__django-11099.tmp-zz00__ab"]
    assert (shard / ".legitimate.tmp-data").is_dir()          # preserved
    assert (shard / "astropy__astropy-7166").is_dir()         # preserved
    assert (shard / ".astropy__astropy-7166.lock").is_file()  # preserved
