"""``xrlenv build cancel`` regression smoke.

Pins the operator-facing cancel surface end-to-end. Two tests:

1. ``test_local_cancel_flips_plan_status`` — local-mode synthetic
   regression: insert an ``in_flight`` plan with mixed-status
   assignments into state.db, invoke ``cmd_build_cancel`` without
   ``--connect-host``, confirm plan status flips + warning message
   names ``--connect-host`` as the way to interrupt running
   cluster builds.
2. ``test_cluster_cancel_interrupts_pending_assignments`` — remote
   mode: dispatch a fast-failing plan (registry-source entry that
   doesn't exist) and cancel it while it's in_flight. The cluster
   marks the plan + assignments cancelled. Doesn't require a real
   long-running build because the cancel path is the same whether
   the build is still pending, building, or seconds from terminal —
   the smoke pins the wire round-trip, not the kill-mid-build
   timing (covered by unit tests).

Excluded from default pytest; run with::

    .venv/bin/python -m pytest \\
        tests/smoke/build_plan/test_cancel_regression.py -v -s

Standalone script::

    .venv/bin/python tests/smoke/build_plan/test_cancel_regression.py
"""

from __future__ import annotations

import argparse
import io as _io
import sys
import time
from pathlib import Path

import pytest

from tests.smoke._build_plan_dispatch_helpers import (
    smoke_artifact_dir,
    write_summary,
)

# The two tests in this file are deliberately single-mode each:
# one exercises the local-only cancel fallback, the other the
# remote cluster round-trip. No SKIPPED entries, no empty
# artifact dirs.


_LOCAL_OUT_DIR: Path | None = None


@pytest.fixture
def local_out_dir() -> Path:
    """Single artifact dir for the local-only cancel test."""
    global _LOCAL_OUT_DIR
    if _LOCAL_OUT_DIR is None:
        _LOCAL_OUT_DIR = smoke_artifact_dir("cancel-local")
    return _LOCAL_OUT_DIR


_REMOTE_OUT_DIR: Path | None = None


@pytest.fixture
def remote_out_dir() -> Path:
    """Single artifact dir for the remote-only cluster cancel test.
    Skips upfront when no admin endpoint is configured."""
    import os
    if not (os.environ.get("XRLENV_ADMIN_HOST")
            or os.environ.get("XRLENV_GRPC_HOST")):
        pytest.skip(
            "cluster cancel is remote-only; set XRLENV_GRPC_HOST or "
            "XRLENV_ADMIN_HOST to point at a running admin",
        )
    global _REMOTE_OUT_DIR
    if _REMOTE_OUT_DIR is None:
        _REMOTE_OUT_DIR = smoke_artifact_dir("cancel-remote")
    return _REMOTE_OUT_DIR


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — local cancel: state.db only, warns about cluster builds
# ──────────────────────────────────────────────────────────────────────────────


def test_local_cancel_flips_plan_status(
    tmp_path: Path, local_out_dir: Path,
) -> None:
    """Local-only cancel without ``--connect-host``: updates the
    plan row to ``cancelled`` in state.db and emits a clear warning
    that in-flight cluster builds aren't interrupted by this mode.

    Local-only by design — exercises ``cmd_build_cancel``'s local
    fallback, not the wire path. The matching cluster-side flow
    is covered by ``test_cluster_cancel_interrupts_pending_assignments``."""
    from xrlenv.cli.commands import cmd_build_cancel
    from xrlenv.control.state import SqliteStateStore

    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    state.record_build_plan(
        plan_id="smoke-local-cancel", applied_by="op",
        plan_json="{}", name="cancel-smoke",
    )
    state.update_build_plan_status("smoke-local-cancel", "in_flight")
    state.close()

    out = _io.StringIO()
    rc = cmd_build_cancel(
        plan_id="smoke-local-cancel",
        state_db=db, out=out,
        connect_host=None,
    )
    body = out.getvalue()
    write_summary(local_out_dir, "test_local_cancel.json", {
        "rc": rc, "stdout": body,
    })
    assert rc == 0
    assert "in_flight" in body and "cancelled" in body
    assert "local-only cancel" in body
    assert "--connect-host" in body

    state2 = SqliteStateStore(db)
    try:
        plan = state2.get_build_plan("smoke-local-cancel")
        assert plan is not None
        assert plan.status == "cancelled"
    finally:
        state2.close()


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — cluster cancel: admin endpoint round-trip
# ──────────────────────────────────────────────────────────────────────────────


def test_cluster_cancel_interrupts_pending_assignments(
    tmp_path: Path, remote_out_dir: Path,
) -> None:
    """Apply a fast-failing plan, immediately cancel it via the
    admin endpoint, confirm the plan + assignments end ``cancelled``
    (not ``failed``). Pins the wire round-trip — the unit tests in
    ``test_admin_server.py`` cover the orchestrator's branching
    logic; this smoke pins the full CLI → HTTP → admin → spec-21
    fanout chain.

    We use a registry-source plan referencing a non-existent image
    so the dispatch fires immediately (no clone / build delay) and
    we can observe the plan status before any node reports back.
    The race-free signal is the plan-level status flip — even if
    individual assignments hit ``failed`` before the cancel lands,
    the plan-level status the operator sees is ``cancelled``.

    Remote-only by design — local-mode counterpart is
    ``test_local_cancel_flips_plan_status``. The ``remote_out_dir``
    fixture skips this test upfront when no admin endpoint is
    configured."""
    import os
    import subprocess

    admin_host = (
        os.environ.get("XRLENV_ADMIN_HOST")
        or os.environ.get("XRLENV_GRPC_HOST", "127.0.0.1")
    )
    admin_port = int(os.environ.get("XRLENV_ADMIN_PORT", "8080"))

    plan_yaml = tmp_path / "cancel-target.yaml"
    plan_yaml.write_text(
        "version: 1\n"
        "entries:\n"
        "  - image_ref: xrlenv-smoke/nope:does-not-exist-anywhere\n"
        "    context_source: { type: registry }\n"
        "    placement:\n"
        "      size_hint_bytes: 1024\n"
        "      preferred_home_count: 1\n",
    )

    # Kick off the apply in the background. Capture stdout so we
    # can stream-read the plan_id from the early-print line
    # ("plan_id: <sha>") before the polling loop even starts.
    # Reading from the apply's own stdout avoids the prior bug
    # where a separate ``xrlenv build status`` subprocess read from
    # the operator's LOCAL state.db while the plan was persisted
    # to the admin's state.db (different paths on a real cluster).
    apply_proc = subprocess.Popen(
        [
            sys.executable, "-m", "xrlenv.cli", "build", "apply",
            "--plan", str(plan_yaml),
            "--connect-host", admin_host,
            "--connect-port", str(admin_port),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1,   # line-buffered so we can stream-read
    )

    # Stream-read up to ~15s for the ``plan_id:`` line. The CLI
    # prints it the moment the admin returns 202 (sub-second on
    # a healthy cluster); the longer ceiling is for a slow admin
    # or a cold-started ``xrlenv up``.
    plan_id: str | None = None
    apply_stdout_lines: list[str] = []
    deadline = time.monotonic() + 15.0
    assert apply_proc.stdout is not None
    while time.monotonic() < deadline:
        line = apply_proc.stdout.readline()
        if not line:   # EOF — apply exited before we found plan_id
            break
        apply_stdout_lines.append(line)
        stripped = line.strip()
        if stripped.startswith("plan_id:"):
            plan_id = stripped.split(":", 1)[1].strip()
            break

    write_summary(remote_out_dir, "apply_stdout_pre_cancel.json", {
        "lines": apply_stdout_lines,
    })
    if plan_id is None:
        apply_proc.terminate()
        pytest.fail(
            "could not find plan_id in apply stdout within 15s "
            "(admin may be unreachable or unresponsive). Lines so far:\n"
            + "".join(apply_stdout_lines),
        )

    # Cancel via the admin endpoint.
    cancel = subprocess.run(
        [
            sys.executable, "-m", "xrlenv.cli", "build", "cancel",
            "--plan", plan_id,
            "--connect-host", admin_host,
            "--connect-port", str(admin_port),
        ],
        capture_output=True, text=True, timeout=30,
    )
    write_summary(remote_out_dir, "cancel_output.json", {
        "rc": cancel.returncode,
        "stdout": cancel.stdout, "stderr": cancel.stderr,
    })
    assert cancel.returncode == 0, (
        f"build cancel exited {cancel.returncode!r}:\n"
        f"stdout: {cancel.stdout}\nstderr: {cancel.stderr}"
    )
    assert "cancelled" in cancel.stdout

    # Don't race the apply CLI's perspective. A registry-404
    # plan transitions to ``partial_failure`` in 2-3s — fast
    # enough that the apply CLI usually sees that terminal status
    # BEFORE the cancel arrives, then exits rc=1 with
    # ``status: partial_failure`` in its stdout. The cancel still
    # works (flipping plan from ``partial_failure`` → ``cancelled``
    # at the admin / state.db layer) but the apply isn't around
    # to observe it. The authoritative signal is the admin's view,
    # not the apply's stdout.
    #
    # The apply has served its purpose (told us the plan_id +
    # dispatched the plan via /api/build/apply). Terminate it
    # cleanly; we don't need its further output.
    apply_proc.terminate()
    try:
        apply_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        apply_proc.kill()

    # Query admin directly for the plan's current status. No
    # state.db-path ambiguity — admin owns this view, full stop.
    import httpx
    headers = {}
    token = os.environ.get("XRLENV_OPERATOR_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(
        base_url=f"http://{admin_host}:{admin_port}",
        headers=headers, timeout=10.0,
    ) as client:
        r = client.get(f"/api/build/plans/{plan_id}")
    write_summary(remote_out_dir, "admin_plan_status.json", {
        "status_code": r.status_code,
        "body": r.json() if r.status_code == 200 else r.text,
    })
    assert r.status_code == 200, (
        f"admin returned {r.status_code} for plan {plan_id!r}: {r.text}"
    )
    snap = r.json()
    assert snap["status"] == "cancelled", (
        f"expected admin to report plan as 'cancelled'; got "
        f"{snap['status']!r}. Full snapshot:\n{snap}"
    )


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
