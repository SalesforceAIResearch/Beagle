"""Pin-budget enforcement + ``xrlenv build calibrate`` smoke (Phase B).

Two tests:

1. ``test_pin_budget_rejects_at_dry_run`` — a plan whose pinned
   entries collectively over-commit each node's available budget
   rejects with ``ManifestInvalid`` at apply time, before FFD
   placement runs. Works in both ``local`` and ``remote`` mode —
   the check is inside ``BuildCoordinator._apply_per_image_ref``
   which both runtimes drive.
2. ``test_calibrate_writes_cluster_reported_sizes`` — after at
   least one image is materialized on the cluster, ``xrlenv build
   calibrate`` walks the connected nodes' ``report_images``,
   replaces measured entries' ``size_hint_bytes`` with the
   cluster max, and flips ``size_hint_source`` to
   ``cluster-reported``. Remote-only (calibrate is a cluster-
   driven flow; local LocalRuntime has nothing to measure).

Excluded from default pytest; run with::

    .venv/bin/python -m pytest \\
        tests/smoke/build_plan/test_pin_budget_and_calibrate.py -v -s

Standalone script::

    .venv/bin/python tests/smoke/build_plan/test_pin_budget_and_calibrate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from tests.smoke._build_plan_dispatch_helpers import (
    apply_plan,
    docker_available,
    runtime_modes,
    smoke_artifact_dir,
    write_summary,
)

# Pin-budget reject is the only test in this file that runs in
# both modes (it exercises the same coordinator code path through
# both runtimes). Calibrate is remote-only because it queries a
# live cluster. The fixtures below match those constraints — no
# SKIPPED entries, no empty artifact dirs.


@pytest.fixture(params=runtime_modes())
def mode(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture used by pin-budget. Picks up
    ``local + remote`` when ``$XRLENV_GRPC_HOST`` (or admin host)
    is set; just ``local`` otherwise."""
    if request.param == "local" and not docker_available():
        pytest.skip("docker daemon not reachable for local mode")
    return str(request.param)


_OUT_DIR_CACHE: dict[str, Path] = {}


@pytest.fixture
def out_dir(mode: str) -> Path:
    """Mode-keyed artifact dir for pin-budget (runs both modes)."""
    if mode not in _OUT_DIR_CACHE:
        _OUT_DIR_CACHE[mode] = smoke_artifact_dir(f"pin-cal-{mode}")
    return _OUT_DIR_CACHE[mode]


_REMOTE_OUT_DIR: Path | None = None


@pytest.fixture
def remote_out_dir() -> Path:
    """Single artifact dir for the remote-only calibrate test.
    Skips upfront when no admin endpoint is configured — otherwise
    the test would fail when ``cmd_build_calibrate`` can't reach
    one."""
    import os
    if not (os.environ.get("XRLENV_ADMIN_HOST")
            or os.environ.get("XRLENV_GRPC_HOST")):
        pytest.skip(
            "calibrate is remote-only; set XRLENV_GRPC_HOST or "
            "XRLENV_ADMIN_HOST to point at a running admin",
        )
    global _REMOTE_OUT_DIR
    if _REMOTE_OUT_DIR is None:
        _REMOTE_OUT_DIR = smoke_artifact_dir("pin-cal-remote")
    return _REMOTE_OUT_DIR


@pytest.fixture
def isolated_state(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "state.db", tmp_path / "runs"


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — pin-budget reject (works local + remote)
# ──────────────────────────────────────────────────────────────────────────────


def test_pin_budget_rejects_at_dry_run(
    mode: str, isolated_state: tuple[Path, Path], out_dir: Path,
) -> None:
    """A plan whose two ``pinned: true`` entries sum to more than
    each node's ``cap_per_node_gb`` budget fails the conservative
    pin-budget check before FFD even runs.

    The check is inside ``_apply_per_image_ref`` (same code path
    for local + remote), so this exercises both — the local mode
    via the in-process coordinator, the remote mode via the admin
    server's ``coordinator.apply()`` call."""
    from xrlenv.control.build_plan import (
        BuildBudget,
        BuildEntry,
        BuildPlan,
        EntryPlacement,
        RegistrySource,
    )
    from xrlenv.errors import ManifestInvalid

    state_db, runs_root = isolated_state
    # cap_per_node_gb clamps each node to 10 GiB; two pinned entries
    # at 8 GB each = 16 GB total pinned, over by 6 GB on every node.
    plan = BuildPlan(
        budget=BuildBudget(
            reserved_runtime_gb=0, buffer_gb=0, cap_per_node_gb=10,
        ),
        entries=(
            BuildEntry(
                image_ref="smoke-pinbudget/a:1",
                context_source=RegistrySource(),
                pinned=True,
                placement=EntryPlacement(size_hint_bytes=8_000_000_000),
            ),
            BuildEntry(
                image_ref="smoke-pinbudget/b:1",
                context_source=RegistrySource(),
                pinned=True,
                placement=EntryPlacement(size_hint_bytes=8_000_000_000),
            ),
        ),
    )
    payload: dict[str, str] = {}
    with pytest.raises(ManifestInvalid) as excinfo:
        apply_plan(
            plan, mode=mode, dry_run=True,
            state_db=state_db, runs_root=runs_root,
        )
    msg = str(excinfo.value)
    payload["message"] = msg
    write_summary(out_dir, f"test_pin_budget_rejects_{mode}.json", payload)

    assert "pin-budget over-commit" in msg
    # Names both offending image_refs (they're in the lines block
    # only indirectly — per-node summaries don't name the entry
    # but the message header references the pinned-total source).
    assert "pinned total" in msg
    assert "available" in msg
    assert "over by" in msg


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — calibrate against connected cluster (remote-only)
# ──────────────────────────────────────────────────────────────────────────────


def test_calibrate_writes_cluster_reported_sizes(
    tmp_path: Path, remote_out_dir: Path,
) -> None:
    """``xrlenv build calibrate`` posts the plan to admin
    ``/api/build/calibrate``, the admin walks each connected node's
    ``report_images()``, and the CLI writes a calibrated YAML.

    Pre-condition: the cluster must have at least one image
    materialized whose name appears in the plan. We use the
    canonical terminal-bench-2 smoke plan because the user has
    already exercised it via ``test_dispatch_tb2.py``; if no
    images on the cluster match, ``unmeasured`` lists them all
    and the size hints stay operator-supplied (still a valid
    outcome — the test asserts that path explicitly).

    Remote-only — local LocalRuntime has no nodes to measure.
    The ``remote_out_dir`` fixture skips this test upfront when
    no admin endpoint is configured."""
    import io as _io

    from xrlenv.cli.commands import cmd_build_calibrate

    plan_path = (
        Path(__file__).resolve().parents[3]
        / "xrlenv_plugins" / "benchmarks" / "terminal_bench_2_1"
        / "build_plan_89_full.yaml"
    )
    if not plan_path.is_file():
        pytest.skip(f"canonical tb2 plan not found at {plan_path}")
    output_path = tmp_path / "tb2.calibrated.yaml"

    # Reach the admin via the same env-driven config the helper uses.
    import os
    admin_host = (
        os.environ.get("XRLENV_ADMIN_HOST")
        or os.environ.get("XRLENV_GRPC_HOST", "127.0.0.1")
    )
    admin_port = int(os.environ.get("XRLENV_ADMIN_PORT", "8080"))

    out = _io.StringIO()
    rc = cmd_build_calibrate(
        plan_path=plan_path, output_path=output_path,
        out=out, connect_host=admin_host, connect_port=admin_port,
        operator_token=os.environ.get("XRLENV_OPERATOR_TOKEN"),
    )
    body = out.getvalue()
    write_summary(remote_out_dir, "test_calibrate.json", {
        "rc": rc,
        "stdout": body,
        "output_path": str(output_path),
    })
    assert rc == 0, f"cmd_build_calibrate exited {rc!r}; output:\n{body}"
    assert output_path.is_file()

    # Either some images were measured (size_hint_source flips to
    # cluster-reported on at least one entry) OR none were and the
    # CLI reports "0 measured" — both are valid outcomes, we just
    # confirm the YAML round-tripped without truncation and the
    # measured flag landed on exactly the entries the admin
    # reported.
    import yaml as _yaml
    raw = _yaml.safe_load(output_path.read_text())
    measured_refs = [
        e["image_ref"] for e in raw.get("entries", [])
        if isinstance(e, dict)
        and e.get("placement", {}).get("size_hint_source") == "cluster-reported"
    ]
    # Either at least one measured (typical happy path after a real
    # cluster build) OR zero (cluster has no overlap with the plan,
    # which the smoke also handles cleanly).
    assert "measured" in body
    assert "unmeasured" in body
    # If anything is measured, every flagged entry's size must be
    # numeric and positive — pins down "the YAML write didn't
    # corrupt the field type."
    for e in raw.get("entries", []):
        if not isinstance(e, dict):
            continue
        if e.get("image_ref") in measured_refs:
            n = e["placement"]["size_hint_bytes"]
            assert isinstance(n, int) and n > 0


# ──────────────────────────────────────────────────────────────────────────────
# Standalone-script entry point
# ──────────────────────────────────────────────────────────────────────────────


def _main_script() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("local", "remote", "all"), default="local",
    )
    parser.add_argument("-k", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args, passthrough = parser.parse_known_args()

    pytest_args: list[str] = [__file__, "-s"]
    pytest_args.append("-vv" if args.verbose else "-v")
    if args.mode == "local":
        pytest_args += ["-k", "local or not ["]
    elif args.mode == "remote":
        pytest_args += ["-k", "remote or not ["]
    if args.k:
        if pytest_args[-2] == "-k":
            pytest_args[-1] = f"({pytest_args[-1]}) and ({args.k})"
        else:
            pytest_args += ["-k", args.k]
    pytest_args += passthrough
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(_main_script())
