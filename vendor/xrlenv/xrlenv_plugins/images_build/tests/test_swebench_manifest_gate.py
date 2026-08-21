"""The legacy images_build SWE-bench build-plan generator must resolve the full 500-id
Verified corpus from the vendored manifest for ``--all`` — never silently fall back to the
8-instance smoke set when swebench isn't importable (audit M11)."""
from __future__ import annotations

import xrlenv_plugins.images_build.swebench_verified.build_plan_gen as legacy


def test_load_verified_instance_ids_returns_full_manifest_not_smoke() -> None:
    ids = legacy._load_verified_instance_ids()
    assert len(ids) == 500
    assert ids != list(legacy.SMOKE_INSTANCES)
    assert ids == sorted(ids) and len(set(ids)) == 500


def test_load_verified_instance_ids_matches_benchmark_local_manifest() -> None:
    # the legacy generator reads the SAME vendored manifest as the canonical benchmark-local
    # generator, so both --all paths plan an identical corpus.
    from pathlib import Path
    manifest = (Path(legacy.__file__).resolve().parents[2] / "benchmarks"
                / "swebench_verified" / "verified_instance_ids.txt")
    vendored = sorted(ln.strip() for ln in manifest.read_text().splitlines()
                      if ln.strip() and not ln.startswith("#"))
    assert legacy._load_verified_instance_ids() == vendored
