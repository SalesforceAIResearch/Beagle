"""Unit tests for the swebench-verified build-plan generator (pure logic; no network).

``probe_sizes=False`` is used throughout to avoid any Docker Hub network calls.
"""
from __future__ import annotations

import pytest
from xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen import (
    DEFAULT_NAMESPACE,
    DEFAULT_SIZE_HINT_BYTES,
    DEFAULT_TAG,
    SMOKE_INSTANCES,
    _instance_to_image_ref,
    generate_plan,
)

# ── _instance_to_image_ref ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("instance_id", "expected"),
    [
        # canonical example from the docstring
        (
            "astropy__astropy-7166",
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-7166:latest",
        ),
        # double underscore at a different position
        (
            "django__django-11099",
            "swebench/sweb.eval.x86_64.django_1776_django-11099:latest",
        ),
        (
            "sympy__sympy-18189",
            "swebench/sweb.eval.x86_64.sympy_1776_sympy-18189:latest",
        ),
        # ensure lowercasing is applied (instance ids are already lower in the dataset,
        # but the docstring specifies lowercasing is part of the transform)
        (
            "Django__Django-9999",
            "swebench/sweb.eval.x86_64.django_1776_django-9999:latest",
        ),
    ],
)
def test_instance_to_image_ref_canonical_transform(instance_id: str, expected: str) -> None:
    assert _instance_to_image_ref(instance_id) == expected


def test_instance_to_image_ref_custom_namespace_and_tag() -> None:
    ref = _instance_to_image_ref("astropy__astropy-7166", namespace="myns", tag="v2")
    assert ref == "myns/sweb.eval.x86_64.astropy_1776_astropy-7166:v2"


def test_instance_to_image_ref_default_namespace_is_swebench() -> None:
    ref = _instance_to_image_ref("astropy__astropy-7166")
    assert ref.startswith(f"{DEFAULT_NAMESPACE}/")


def test_instance_to_image_ref_default_tag_is_latest() -> None:
    ref = _instance_to_image_ref("astropy__astropy-7166")
    assert ref.endswith(f":{DEFAULT_TAG}")


# ── generate_plan ─────────────────────────────────────────────────────────────


def test_generate_plan_one_instance_context_source_is_registry() -> None:
    plan = generate_plan(["astropy__astropy-7166"], probe_sizes=False)
    assert len(plan["entries"]) == 1
    entry = plan["entries"][0]
    assert entry["context_source"] == {"type": "registry"}


def test_generate_plan_image_ref_matches_instance_to_image_ref() -> None:
    instance_id = "astropy__astropy-7166"
    plan = generate_plan([instance_id], probe_sizes=False)
    entry = plan["entries"][0]
    assert entry["image_ref"] == _instance_to_image_ref(instance_id)


def test_generate_plan_labels_contain_instance_id() -> None:
    instance_id = "django__django-11099"
    plan = generate_plan([instance_id], probe_sizes=False)
    entry = plan["entries"][0]
    assert entry["labels"]["xrlenv.instance_id"] == instance_id


def test_generate_plan_labels_contain_benchmark_name() -> None:
    plan = generate_plan(["astropy__astropy-7166"], probe_sizes=False)
    entry = plan["entries"][0]
    assert entry["labels"]["xrlenv.benchmark"] == "swebench-verified"


def test_generate_plan_uses_default_size_hint_when_no_probe() -> None:
    plan = generate_plan(["astropy__astropy-7166"], probe_sizes=False)
    entry = plan["entries"][0]
    assert entry["placement"]["size_hint_bytes"] == DEFAULT_SIZE_HINT_BYTES
    assert entry["placement"]["size_hint_source"] == "heuristic"


def test_generate_plan_multiple_instances_produces_correct_count() -> None:
    instances = ["astropy__astropy-7166", "django__django-11099", "sympy__sympy-18189"]
    plan = generate_plan(instances, probe_sizes=False)
    assert len(plan["entries"]) == 3


def test_generate_plan_entry_order_matches_instance_order() -> None:
    instances = ["sympy__sympy-18189", "django__django-11099", "astropy__astropy-7166"]
    plan = generate_plan(instances, probe_sizes=False)
    for i, inst in enumerate(instances):
        assert plan["entries"][i]["labels"]["xrlenv.instance_id"] == inst


def test_generate_plan_top_level_shape() -> None:
    plan = generate_plan(["astropy__astropy-7166"], probe_sizes=False)
    assert plan["version"] == 1
    assert "name" in plan
    assert "replication" in plan
    assert "budget" in plan
    assert "entries" in plan


def test_generate_plan_smoke_set_name_is_smoke8() -> None:
    """When exactly the SMOKE_INSTANCES list is passed, the plan name uses 'smoke-8'."""
    plan = generate_plan(list(SMOKE_INSTANCES), probe_sizes=False)
    assert "smoke-8" in plan["name"]


def test_generate_plan_non_smoke_set_name_includes_count() -> None:
    instances = ["astropy__astropy-7166", "django__django-11099"]
    plan = generate_plan(instances, probe_sizes=False)
    assert "2-instance" in plan["name"]


def test_generate_plan_custom_namespace_reflected_in_image_ref() -> None:
    plan = generate_plan(
        ["astropy__astropy-7166"], namespace="myns", probe_sizes=False,
    )
    assert plan["entries"][0]["image_ref"].startswith("myns/")


def test_generate_plan_custom_tag_reflected_in_image_ref() -> None:
    plan = generate_plan(
        ["astropy__astropy-7166"], tag="v99", probe_sizes=False,
    )
    assert plan["entries"][0]["image_ref"].endswith(":v99")


def test_generate_plan_pinned_false_by_default() -> None:
    plan = generate_plan(["astropy__astropy-7166"], probe_sizes=False)
    assert plan["entries"][0]["pinned"] is False


def test_generate_plan_pinned_true_when_requested() -> None:
    plan = generate_plan(["astropy__astropy-7166"], pinned=True, probe_sizes=False)
    assert plan["entries"][0]["pinned"] is True


# ── M6: --all must not silently plan a partial cache ──────────────────────────


def _partial_shard(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> object:
    # A genuine partial: authoritative = {a, b}, the requested corpus is the SUBSET {a} —
    # a MISSING id, no extras, no dups. (No count-only degradation is exercised: audit M11
    # made _authoritative_ids manifest-only, so a valid authority is always available.)
    import xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen as bpg
    monkeypatch.setattr(bpg, "_authoritative_ids", lambda: {"a", "b"})
    monkeypatch.setattr(bpg, "_load_verified_instance_ids", lambda: ["a"])
    return bpg


def test_all_fails_closed_on_partial_cache(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # audit M6: --all must FAIL closed (nonzero), not warn + emit a partial plan.
    bpg = _partial_shard(tmp_path, monkeypatch)
    rc = bpg.main(["--all", "--no-probe", "--output", "-"])  # type: ignore[attr-defined]
    assert rc == 1
    assert "ERROR" in capsys.readouterr().err


def test_all_partial_allowed_with_explicit_flag(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # --allow-partial relaxes a MISSING subset -> warn + proceed.
    bpg = _partial_shard(tmp_path, monkeypatch)
    rc = bpg.main(["--all", "--allow-partial", "--no-probe", "--output", "-"])  # type: ignore[attr-defined]
    assert rc in (0, None)
    assert "WARNING" in capsys.readouterr().err


def test_all_fails_closed_when_manifest_unavailable(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # audit M11: with NO validated authority, --all must fail closed — even under
    # --allow-partial (which relaxes a missing subset, NOT the absence of an authority). The
    # old count-only degradation ("accept any 500 unique ids") is removed.
    import xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen as bpg
    monkeypatch.setattr(bpg, "_authoritative_ids", lambda: None)
    monkeypatch.setattr(bpg, "_load_verified_instance_ids", lambda: ["a", "b"])
    rc = bpg.main(["--all", "--allow-partial", "--no-probe", "--output", "-"])
    assert rc == 1
    assert "missing/invalid" in capsys.readouterr().err


def test_all_allow_partial_still_rejects_unexpected_ids(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # audit Low: --allow-partial permits a SUBSET (missing ids) but NOT ids outside the
    # authoritative corpus — an unexpected id is never an intentional subset.
    import xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen as bpg
    monkeypatch.setattr(bpg, "_authoritative_ids", lambda: {"a", "b"})
    monkeypatch.setattr(bpg, "_load_verified_instance_ids", lambda: ["a", "X"])  # X unexpected
    rc = bpg.main(["--all", "--allow-partial", "--no-probe", "--output", "-"])
    assert rc == 1
    assert "NOT in the authoritative Verified corpus" in capsys.readouterr().err


def test_all_fails_on_wrong_membership_even_with_right_count(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # audit M6: WRONG ids must fail — membership, not just count (here count 2 == 2).
    import xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen as bpg
    monkeypatch.setattr(bpg, "_authoritative_ids", lambda: {"good-a", "good-b"})
    monkeypatch.setattr(bpg, "_load_verified_instance_ids", lambda: ["good-a", "WRONG-b"])
    rc = bpg.main(["--all", "--no-probe", "--output", "-"])
    assert rc == 1
    assert "authoritative Verified corpus" in capsys.readouterr().err


def test_authoritative_ids_reads_vendored_manifest_offline() -> None:
    # audit H4: the membership gate must NOT degrade to count-only — the vendored 500-id
    # manifest makes _authoritative_ids offline-reliable (never None just because the
    # dataset loader can't reach the network).
    import xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen as bpg
    auth = bpg._authoritative_ids()
    assert auth is not None
    assert len(auth) == 500


def test_all_fails_on_duplicate_id_even_when_membership_ok(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # audit Low: 500 authoritative ids PLUS a duplicate passes membership but emits 501
    # entries — exact cardinality must fail it.
    import xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen as bpg
    monkeypatch.setattr(bpg, "_authoritative_ids", lambda: {"a", "b"})
    monkeypatch.setattr(bpg, "_load_verified_instance_ids", lambda: ["a", "b", "b"])
    rc = bpg.main(["--all", "--no-probe", "--output", "-"])
    assert rc == 1
    assert "duplicate" in capsys.readouterr().err


# ── M11: manifest integrity is enforced, not just read ────────────────────────


def test_read_verified_manifest_intact_returns_500() -> None:
    import xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen as bpg
    ids = bpg.read_verified_manifest()
    assert ids is not None
    assert len(ids) == 500 and ids == sorted(ids) and len(set(ids)) == 500


def test_read_verified_manifest_rejects_wrong_count(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen as bpg
    good = bpg.read_verified_manifest()
    assert good is not None
    bad = tmp_path / "m499.txt"  # type: ignore[operator]
    bad.write_text("\n".join(good[:-1]) + "\n")     # 499 ids -> count mismatch
    monkeypatch.setattr(bpg, "_MANIFEST", bad)
    assert bpg.read_verified_manifest() is None


def test_read_verified_manifest_rejects_bad_digest(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen as bpg
    good = bpg.read_verified_manifest()
    assert good is not None
    bad = tmp_path / "mbad.txt"  # type: ignore[operator]
    bad.write_text("# sha256(ids) : deadbeef\n" + "\n".join(good) + "\n")  # right count, wrong digest
    monkeypatch.setattr(bpg, "_MANIFEST", bad)
    assert bpg.read_verified_manifest() is None


def test_load_verified_instance_ids_prefers_complete_cache(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # audit Low: --all sources ONLY complete cache entries — a bare {} anchor dir is excluded.
    import xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen as bpg
    from xrlenv_plugins.benchmarks.swebench_verified.build_cache import _row_to_cache
    shard = tmp_path / "swebench-verified"  # type: ignore[operator]
    good = shard / "astropy__astropy-7166"
    good.mkdir(parents=True)
    for name, text in _row_to_cache(
        {"instance_id": "astropy__astropy-7166", "patch": "p", "problem_statement": "s"},
    ).items():
        (good / name).write_text(text)
    bare = shard / "django__django-11099"
    bare.mkdir(parents=True)
    (bare / "instance.json").write_text("{}")            # incomplete -> excluded
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(tmp_path))
    assert bpg._load_verified_instance_ids() == ["astropy__astropy-7166"]


def test_load_verified_instance_ids_falls_back_to_manifest_never_smoke(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # audit Low: with NO complete cache, --all sources the VALIDATED 500-id manifest, never the
    # 8-instance smoke set (the removed implicit fallback).
    import xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen as bpg
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(tmp_path))   # empty cache
    ids = bpg._load_verified_instance_ids()
    assert len(ids) == 500 and ids != list(bpg.SMOKE_INSTANCES)


def test_authoritative_mismatch_fails_closed_without_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # audit Low: the helper no longer count-only-accepts when authority is unavailable — it
    # returns a mismatch reason (fail closed) for any direct caller.
    import xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen as bpg
    monkeypatch.setattr(bpg, "_authoritative_ids", lambda: None)
    reason = bpg._authoritative_mismatch(["a"] * bpg.VERIFIED_TOTAL)   # right COUNT
    assert reason is not None and "unavailable/invalid" in reason


def test_all_allow_partial_still_rejects_duplicates(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # audit Low: --allow-partial relaxes MEMBERSHIP (a subset) but must still reject DUPLICATE
    # ids (repeated image entries are never intentional).
    import xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen as bpg
    monkeypatch.setattr(bpg, "_authoritative_ids", lambda: {"a", "b"})
    monkeypatch.setattr(bpg, "_load_verified_instance_ids", lambda: ["a", "b", "b"])
    rc = bpg.main(["--all", "--allow-partial", "--no-probe", "--output", "-"])
    assert rc == 1
    assert "duplicate" in capsys.readouterr().err
