"""Build-plan dispatch smoke for seta-env.

seta-env's Harbor-Dataset publishes Dockerfiles only — no prebuilt
registry images — so every entry in the canonical seta-env plan
uses ``context_source: type: git``. Source-build dispatch (clone +
``docker build``) is live for both git AND tarball entries
(sub-slice 1.b). The remaining boundary contract: programmatic
callers that build a plan in-memory and skip the CLI's
``resolve_tarball_sources`` helper get a clear apply-time
rejection pointing at the helper.

Three tests:

1. The committed seta-env starter plan loads and parses. Every
   entry is a ``GitSource`` with the expected upstream repo +
   per-task subdir.
2. ``dry_run=True`` apply against the seta-env starter plan
   reaches the placement layer without source-type rejection.
   Confirms git-source entries dispatch through the coordinator's
   build_image_fn path. Doesn't actually build anything.
3. A synthetic tarball-source plan WITHOUT ``content_b64`` is
   rejected at apply time with a message pointing at
   ``resolve_tarball_sources``.

A real cluster apply of the seta-env starter (clone + build all
16 entries, ~30+ min wall-clock + real network) is operator-
driven via ``xrlenv build apply --plan
xrlenv_plugins/benchmarks/seta/build_plan.yaml``; not
automated here because the cost is high enough that re-running
it casually wastes time and bandwidth.

Excluded from the default ``pytest -q`` suite via ``addopts =
"--ignore=tests/smoke"`` in ``pyproject.toml``. Run explicitly:

pytest mode::

    .venv/bin/python -m pytest \\
        tests/smoke/test_build_plan_dispatch_seta_env.py -v -s

script mode::

    .venv/bin/python tests/smoke/test_build_plan_dispatch_seta_env.py

See ``tests/smoke/README.md`` for the full runbook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
from xrlenv.control.build_plan import BuildPlan, GitSource, load_build_plan
from xrlenv.errors import ManifestInvalid

from tests.smoke._build_plan_dispatch_helpers import (
    apply_plan,
    docker_available,
    runtime_modes,
    smoke_artifact_dir,
    write_summary,
)

CANONICAL_PLAN_PATH = (
    # this file lives at tests/smoke/build_plan/<file>.py;
    # project root is 4 parents up.
    Path(__file__).resolve().parents[3]
    / "xrlenv_plugins" / "benchmarks" / "seta" / "build_plan.yaml"
)


@pytest.fixture(params=runtime_modes())
def mode(request: pytest.FixtureRequest) -> str:
    if request.param == "local" and not docker_available():
        pytest.skip("docker daemon not reachable for local mode")
    return str(request.param)


@pytest.fixture(scope="module")
def canonical_plan() -> BuildPlan:
    plan = load_build_plan(CANONICAL_PLAN_PATH)
    assert plan.is_per_image_ref(), "seta-env canonical plan must be entries-shaped"
    return plan


_OUT_DIR_CACHE: dict[str, Path] = {}


@pytest.fixture
def out_dir(mode: str) -> Path:
    """One artifact dir per ``(pytest invocation, mode)`` pair,
    cached at module load."""
    if mode not in _OUT_DIR_CACHE:
        _OUT_DIR_CACHE[mode] = smoke_artifact_dir(f"seta-env-{mode}")
    return _OUT_DIR_CACHE[mode]


@pytest.fixture
def isolated_state(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "state.db", tmp_path / "runs"


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — load + structural assertions
# ──────────────────────────────────────────────────────────────────────────────


def test_canonical_plan_loads(canonical_plan: BuildPlan) -> None:
    assert len(canonical_plan.entries) >= 1
    for e in canonical_plan.entries:
        assert isinstance(e.context_source, GitSource), (
            f"{e.image_ref}: expected GitSource, got "
            f"{type(e.context_source).__name__}"
        )
        cs = e.context_source
        assert cs.repo == "https://github.com/camel-ai/seta-env"
        assert cs.subdir.startswith("Harbor-Dataset/")
        assert cs.subdir.endswith("/environment")
        assert cs.dockerfile == "Dockerfile"
        assert e.labels.get("xrlenv.benchmark") == "seta-env"


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — git-source dry-run reaches placement (no rejection)
# ──────────────────────────────────────────────────────────────────────────────


def test_git_source_dry_run_reaches_placement(
    mode: str, canonical_plan: BuildPlan,
    isolated_state: tuple[Path, Path], out_dir: Path,
) -> None:
    """Confirms the coordinator's source-type gate no longer rejects
    git-source entries. ``dry_run=True`` validates + plans placement
    but doesn't dispatch — useful as a fast sanity check that the
    plan is otherwise well-formed before committing to a real
    multi-minute build."""
    state_db, runs_root = isolated_state
    result = apply_plan(
        canonical_plan, mode=mode, dry_run=True,
        state_db=state_db, runs_root=runs_root,
    )
    write_summary(out_dir, "test_git_source_dry_run.json", result.raw)
    assert result.status == "dry_run", (
        f"expected dry_run status, got {result.status!r}"
    )
    # Placement happened — every entry got a per-(node, image_ref) row.
    # Local mode serializes BuildOutcome.placement as a dict with
    # ``assignments`` (list of {image_ref, node_id, ...}). Remote
    # dry-run returns the same shape under the top-level
    # ``placement`` key but with positional assignments.
    placement = result.raw.get("placement")
    if placement is not None:
        if isinstance(placement, dict):
            assignments = placement.get("assignments") or []
        elif isinstance(placement, list):
            assignments = placement
        else:
            assignments = []
        placed_refs = {a["image_ref"] for a in assignments}
        plan_refs = {e.image_ref for e in canonical_plan.entries}
        assert placed_refs == plan_refs, (
            f"dry-run placement should cover every entry; "
            f"missing: {plan_refs - placed_refs}, extra: {placed_refs - plan_refs}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — unresolved tarball entries reject with the right hint
#
# Sub-slice 1.b shipped tarball dispatch (operator's CLI calls
# ``resolve_tarball_sources`` immediately after YAML load). The
# remaining boundary contract: a programmatic caller that builds a
# BuildPlan in-memory and skips that helper still gets a clear
# rejection at apply time pointing at the helper, instead of
# silently shipping a wire payload missing its bytes.
# ──────────────────────────────────────────────────────────────────────────────


def test_apply_rejects_unresolved_tarball_with_operator_friendly_error(
    mode: str, isolated_state: tuple[Path, Path], out_dir: Path,
) -> None:
    """Synthetic tarball plan WITHOUT ``content_b64`` exercises the
    "you forgot to call resolve_tarball_sources" guard."""
    from xrlenv.control.build_plan import (
        BuildEntry,
        EntryPlacement,
        TarballSource,
    )

    state_db, runs_root = isolated_state
    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="my-org/private-task:v3",
            context_source=TarballSource(
                path="./contexts/private-task.tar.gz",
                dockerfile="Dockerfile",
                # NOTE: no content_b64 — bypasses the CLI helper.
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    error_payload: dict[str, str] = {}
    with pytest.raises(ManifestInvalid) as excinfo:
        apply_plan(
            plan, mode=mode, dry_run=True,
            state_db=state_db, runs_root=runs_root,
        )
    msg = str(excinfo.value)
    error_payload["message"] = msg
    write_summary(out_dir, "test_apply_rejects_unresolved_tarball.json", error_payload)

    assert "resolve_tarball_sources" in msg, (
        f"rejection message should point at the CLI helper "
        f"that resolves bytes; got: {msg!r}"
    )
    assert "my-org/private-task:v3" in msg, (
        f"rejection message should name the offending image_ref; "
        f"got: {msg!r}"
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
    parser.add_argument("-k", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args, passthrough = parser.parse_known_args()

    pytest_args: list[str] = [__file__, "-s"]
    pytest_args.append("-vv" if args.verbose else "-v")
    # ``not [`` matches tests with no parametrize bracket — i.e.
    # mode-agnostic tests like ``test_canonical_plan_loads``. Without
    # this clause a bare ``-k local`` filter would deselect them
    # entirely, which is misleading: the script defaulting to
    # ``--mode local`` shouldn't quietly skip the non-mode tests.
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
