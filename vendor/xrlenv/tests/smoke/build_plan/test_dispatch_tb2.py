"""Build-plan dispatch smoke for terminal-bench-2.

Five tests against a real Docker daemon (or a remote control plane
when ``XRLENV_GRPC_HOST`` / ``XRLENV_ADMIN_HOST`` is set):

1. Apply a generated phase-0 plan (8 SMOKE_TASKS; each entry's
   ref read from the populated shard's ``task.toml``).
2. Idempotent re-apply — ``no_op_already_completed`` short-circuit.
3. ``--force`` re-apply — re-dispatches all 8 entries.
4. Calibration: hint vs measured uncompressed size, with a soft
   bound (1x ≤ ratio ≤ 5x). Local mode only — needs
   ``docker image inspect`` access on the same host the test runs.
5. Fresh-8 dispatch — generate a plan with 8 non-smoke task ids
   that aren't pulled locally yet, apply it, verify all reach
   ``done``.

Excluded from the default ``pytest -q`` suite via ``addopts =
"--ignore=tests/smoke"`` in ``pyproject.toml``. Run explicitly:

pytest mode (both local + remote if remote configured)::

    .venv/bin/python -m pytest \\
        tests/smoke/test_build_plan_dispatch_tb2.py -v -s

script mode (local only)::

    .venv/bin/python tests/smoke/test_build_plan_dispatch_tb2.py

script mode (remote — requires $XRLENV_GRPC_HOST or
$XRLENV_ADMIN_HOST + $XRLENV_OPERATOR_TOKEN)::

    XRLENV_GRPC_HOST=10.0.0.10 \\
    XRLENV_OPERATOR_TOKEN=$(cat ~/.xrlenv/secrets/operator.token) \\
    .venv/bin/python tests/smoke/test_build_plan_dispatch_tb2.py --mode remote

See ``tests/smoke/README.md`` for the full runbook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
from xrlenv.control.build_plan import BuildPlan

from tests.smoke._build_plan_dispatch_helpers import (
    ApplyResult,
    apply_plan,
    collect_calibration_rows,
    docker_available,
    format_calibration_table,
    image_present_locally,
    local_image_created_at,
    pick_fresh_tb2_tasks,
    runtime_modes,
    smoke_artifact_dir,
    write_calibrated_plan,
    write_summary,
)

SMOKE_TASKS = (
    "fix-git", "build-pov-ray", "overfull-hbox", "cobol-modernization",
    "prove-plus-comm", "constraints-scheduling", "nginx-request-logging",
    "dna-insert",
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(params=runtime_modes())
def mode(request: pytest.FixtureRequest) -> str:
    """Dispatch mode: ``local`` (always) or ``remote`` (when
    ``$XRLENV_GRPC_HOST`` or ``$XRLENV_ADMIN_HOST`` is set)."""
    if request.param == "local" and not docker_available():
        pytest.skip("docker daemon not reachable for local mode")
    return str(request.param)


@pytest.fixture(scope="module")
def canonical_plan() -> BuildPlan:
    """A terminal-bench-2-1 phase-0 build plan (8 SMOKE_TASKS),
    generated from the populated shard's authoritative per-task
    ``docker_image`` refs. Uses registry-probe sizes so the
    calibration test's hint-vs-actual band stays meaningful. Skips
    if the shard isn't populated with all 8 smoke tasks."""
    from xrlenv_plugins.benchmarks.terminal_bench_2_1.build_plan_gen import (
        _discover_all_tasks,
        _shard_dir,
        generate_plan,
    )

    shard = _shard_dir()
    present = set(_discover_all_tasks(shard))
    missing = [t for t in SMOKE_TASKS if t not in present]
    if missing:
        pytest.skip(
            f"terminal-bench-2-1 shard at {shard} missing smoke task(s) "
            f"{missing!r}; populate it (build_cache.py --stage populate) "
            "or set $XRLENV_BENCHMARK_CACHE.",
        )
    plan_dict = generate_plan(list(SMOKE_TASKS), shard_dir=shard)
    plan = BuildPlan.model_validate(plan_dict)
    assert plan.is_per_image_ref(), "tb2 canonical plan must be entries-shaped"
    assert len(plan.entries) == 8
    return plan


_OUT_DIR_CACHE: dict[str, Path] = {}


@pytest.fixture
def out_dir(mode: str) -> Path:
    """One artifact dir per ``(pytest invocation, mode)`` pair,
    cached at module load. Without the cache each function-scoped
    test would land in its own ``tmp/smoke-build-plan-tb2-<mode>-<ts>/``,
    making the calibration side-artifact (and everything else) hard
    to find. With the cache: ``[local]`` runs share one dir,
    ``[remote]`` runs share another, and a single ``--mode all``
    invocation produces exactly two artifact dirs total."""
    if mode not in _OUT_DIR_CACHE:
        _OUT_DIR_CACHE[mode] = smoke_artifact_dir(f"tb2-{mode}")
    return _OUT_DIR_CACHE[mode]


@pytest.fixture
def isolated_state(tmp_path: Path, mode: str) -> tuple[Path, Path]:
    """Per-test state.db + runs root for local mode. Remote mode
    targets the cluster's persistent state.db; ``tmp_path`` is
    unused there."""
    state_db = tmp_path / "state.db"
    runs_root = tmp_path / "runs"
    return state_db, runs_root


def _assert_every_entry_done(
    plan: BuildPlan, result: ApplyResult, *, mode: str,
) -> None:
    """Verify exactly len(plan.entries) assignments reached ``done``
    AND no assignment failed.

    The strict count check holds on both modes because the
    coordinator's force re-apply path now purges prior assignments
    for the plan_id before recording the new placement (otherwise
    stale rows from earlier applies on now-vacated nodes would
    inflate the count when the bin-packer chooses different nodes
    across runs).

    Remote mode adds a belt-and-suspenders per-entry verification:
    walk ``result.raw['assignments']`` and confirm every plan
    entry's ``image_ref`` has at least one ``done`` row. Catches
    subtle bugs where the count is right but the wrong refs
    materialized (e.g. placement contamination across plans).
    """
    assert result.failures == 0, (
        f"apply reported failures: {result.error_summary}"
    )
    assert result.successes == len(plan.entries), (
        f"expected {len(plan.entries)} successes, got {result.successes}; "
        f"if successes > expected, a prior coordinator change may have "
        f"regressed the force-reapply purge in _apply_per_image_ref."
    )
    if mode == "remote":
        assignments = result.raw.get("assignments") or []
        done_refs = {
            a["image_ref"] for a in assignments if a.get("status") == "done"
        }
        missing = [
            e.image_ref for e in plan.entries if e.image_ref not in done_refs
        ]
        assert not missing, (
            f"these plan entries did not reach done: {missing!r}; "
            f"assignments: {assignments}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — apply the committed phase-0 plan
# ──────────────────────────────────────────────────────────────────────────────


def test_apply_canonical_plan(
    mode: str, canonical_plan: BuildPlan,
    isolated_state: tuple[Path, Path], out_dir: Path,
) -> None:
    state_db, runs_root = isolated_state
    # force=True so the test is robust to whichever state.db state
    # the cluster's already in. In local mode tmp_path makes state.db
    # fresh per-test; in remote mode the cluster's persistent state.db
    # may already have this plan_id as completed from a prior run, in
    # which case a no-force apply would short-circuit to
    # no_op_already_completed and skip the actual dispatch we want
    # to exercise. test_idempotent_reapply is where no-force semantics
    # are actually validated.
    result = apply_plan(
        canonical_plan, mode=mode, force=True,
        state_db=state_db, runs_root=runs_root,
    )
    write_summary(out_dir, "test_apply_canonical_plan.json", result.raw)

    assert result.status == "completed", (
        f"apply did not complete: status={result.status!r}, "
        f"errors={result.error_summary}"
    )
    _assert_every_entry_done(canonical_plan, result, mode=mode)

    if mode == "local":
        # Every smoke image is now in `docker images`.
        for ref in (e.image_ref for e in canonical_plan.entries):
            assert image_present_locally(ref), (
                f"expected {ref} in `docker images` after apply"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — idempotent re-apply (no_op_already_completed)
# ──────────────────────────────────────────────────────────────────────────────


def test_idempotent_reapply(
    mode: str, canonical_plan: BuildPlan,
    isolated_state: tuple[Path, Path], out_dir: Path,
) -> None:
    state_db, runs_root = isolated_state
    # force=True on the baseline so the test is robust to cluster
    # state.db state. In local mode tmp_path makes state.db fresh
    # per-test so plain apply works; in remote mode the cluster's
    # persistent state.db sees the canonical plan as already
    # completed from any previous test 1 run, which would short-
    # circuit this baseline to no_op_already_completed and break
    # the contract we're trying to test.
    first = apply_plan(
        canonical_plan, mode=mode, force=True,
        state_db=state_db, runs_root=runs_root,
    )
    assert first.status == "completed"

    pre_created_at: dict[str, str | None] = {}
    if mode == "local":
        pre_created_at = {
            e.image_ref: local_image_created_at(e.image_ref)
            for e in canonical_plan.entries
        }

    second = apply_plan(
        canonical_plan, mode=mode,
        state_db=state_db, runs_root=runs_root,
    )
    write_summary(out_dir, "test_idempotent_reapply.json", {
        "first": first.raw, "second": second.raw,
    })
    assert second.status == "no_op_already_completed", (
        f"re-apply not short-circuited: status={second.status!r}"
    )

    if mode == "local":
        # No image was re-pulled; Docker's Created timestamp must
        # match the first-apply value (Docker only updates Created
        # on a real pull, not on a no-op cache hit).
        for ref, before in pre_created_at.items():
            after = local_image_created_at(ref)
            assert before == after, (
                f"{ref}: Created timestamp moved from {before!r} to "
                f"{after!r} — re-pull happened despite the no-op short-circuit"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — --force re-apply
# ──────────────────────────────────────────────────────────────────────────────


def test_force_reapply(
    mode: str, canonical_plan: BuildPlan,
    isolated_state: tuple[Path, Path], out_dir: Path,
) -> None:
    state_db, runs_root = isolated_state
    # force=True on the baseline (see test_idempotent_reapply for
    # rationale): keeps remote-mode runs robust to whatever state
    # the cluster's persistent state.db is already in.
    first = apply_plan(
        canonical_plan, mode=mode, force=True,
        state_db=state_db, runs_root=runs_root,
    )
    assert first.status == "completed"

    forced = apply_plan(
        canonical_plan, mode=mode, force=True,
        state_db=state_db, runs_root=runs_root,
    )
    write_summary(out_dir, "test_force_reapply.json", forced.raw)
    assert forced.status == "completed", (
        f"--force apply did not complete: status={forced.status!r}, "
        f"errors={forced.error_summary}"
    )
    _assert_every_entry_done(canonical_plan, forced, mode=mode)


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — calibration (local only)
# ──────────────────────────────────────────────────────────────────────────────


def test_calibration_hint_vs_actual(
    mode: str, canonical_plan: BuildPlan,
    isolated_state: tuple[Path, Path], out_dir: Path,
) -> None:
    if mode != "local":
        pytest.skip(
            "calibration needs `docker image inspect` access; "
            "calibration is local-only — runs against the same host's "
            "Docker daemon, not a remote cluster",
        )
    state_db, runs_root = isolated_state
    result = apply_plan(
        canonical_plan, mode=mode,
        state_db=state_db, runs_root=runs_root,
    )
    assert result.status == "completed"

    rows = collect_calibration_rows(canonical_plan)
    table = format_calibration_table(rows)
    print()
    print(table)

    (out_dir / "calibration.txt").write_text(table)
    write_calibrated_plan(
        canonical_plan, rows, out_dir / "build_plan.calibrated.yaml",
    )

    # Soft bound: hint is the registry-probe compressed size; actual
    # is the on-disk uncompressed size. ratio = actual/hint should
    # be roughly 1x-3x for typical OCI images. Catch order-of-magnitude
    # misestimates without being flaky.
    for r in rows:
        assert r.actual_bytes is not None, (
            f"{r.image_ref}: image absent after apply (calibration row missing)"
        )
        assert r.ratio is not None
        assert 1.0 <= r.ratio <= 5.0, (
            f"{r.image_ref}: actual/hint ratio = {r.ratio:.2f} "
            f"(hint={r.hint_bytes}, actual={r.actual_bytes}); "
            "outside the expected 1x..5x band — registry-probe may have "
            "regressed or the image structure changed dramatically"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — fresh-8 dispatch (genuinely-uncached images)
# ──────────────────────────────────────────────────────────────────────────────


def test_fresh_eight_dispatch(
    mode: str, isolated_state: tuple[Path, Path], out_dir: Path,
) -> None:
    from xrlenv_plugins.benchmarks.terminal_bench_2_1.build_plan_gen import (
        _shard_dir,
        generate_plan,
    )

    state_db, runs_root = isolated_state

    # Pre-flight: pick 8 tb2 task ids that are NOT in the smoke set
    # AND aren't already pulled locally (when in local mode).
    fresh = pick_fresh_tb2_tasks(
        n=8, exclude=SMOKE_TASKS,
        require_uncached=(mode == "local"),
    )
    # offline-friendly (heuristic sizes); refs read per-task from task.toml.
    plan_dict = generate_plan(fresh, shard_dir=_shard_dir(), probe_sizes=False)
    plan = BuildPlan.model_validate(plan_dict)
    assert plan.is_per_image_ref()
    assert len(plan.entries) == 8

    # force=True for the same robustness rationale as test 1 — on
    # remote mode this plan_id may have been seen before if a prior
    # run picked the same 8 task ids (the require_uncached filter
    # only applies in local mode).
    result = apply_plan(
        plan, mode=mode, force=True,
        state_db=state_db, runs_root=runs_root,
    )
    write_summary(out_dir, "test_fresh_eight_dispatch.json", {
        "task_ids": fresh, "result": result.raw,
    })

    assert result.status == "completed", (
        f"fresh-8 apply did not complete: status={result.status!r}, "
        f"errors={result.error_summary}"
    )
    _assert_every_entry_done(plan, result, mode=mode)

    if mode == "local":
        # refs come from each task's task.toml (mixed upstream tags),
        # so read them from the generated plan rather than synthesizing.
        for ref in (e.image_ref for e in plan.entries):
            assert image_present_locally(ref), (
                f"expected {ref} in `docker images` after fresh-8 apply"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Standalone-script entry point
# ──────────────────────────────────────────────────────────────────────────────


def _main_script() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("local", "remote", "all"), default="local",
        help="Run against the local Docker daemon, a remote cluster, or both.",
    )
    parser.add_argument(
        "-k", default=None,
        help="pytest -k expression (filter tests by name).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose pytest output (default on).",
    )
    args, passthrough = parser.parse_known_args()

    pytest_args: list[str] = [__file__, "-s"]
    pytest_args.append("-vv" if args.verbose else "-v")
    # ``not [`` matches tests with no parametrize bracket — i.e.
    # mode-agnostic tests. Without this clause a bare ``-k local``
    # filter would deselect them entirely, which is misleading: the
    # script defaulting to ``--mode local`` shouldn't quietly skip
    # the non-mode tests.
    if args.mode == "local":
        pytest_args += ["-k", "local or not ["]
    elif args.mode == "remote":
        pytest_args += ["-k", "remote or not ["]
    if args.k:
        pytest_args[-1] = f"({pytest_args[-1]}) and ({args.k})" \
            if pytest_args[-2] == "-k" else args.k
    pytest_args += passthrough
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(_main_script())
