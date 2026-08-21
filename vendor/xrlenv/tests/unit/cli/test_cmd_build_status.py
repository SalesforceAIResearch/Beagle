"""``xrlenv build status`` CLI (P1.6.c)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from xrlenv.cli.commands import cmd_build_apply, cmd_build_status
from xrlenv.control.state import BuildAssignmentRecord, SqliteStateStore


def test_status_with_no_plans_says_so(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    SqliteStateStore(db).close()
    out = io.StringIO()
    rc = cmd_build_status(plan_id=None, state_db=db, out=out)
    assert rc == 0
    assert "no build plans" in out.getvalue()


def test_status_picks_most_recent_plan(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    state.record_build_plan(
        plan_id="aaa", applied_by="cli", plan_json="{}",
    )
    state.update_build_plan_status("aaa", "completed")
    state.record_build_plan(
        plan_id="bbb", applied_by="cli", plan_json="{}",
    )
    state.update_build_plan_status("bbb", "completed")
    state.close()
    out = io.StringIO()
    rc = cmd_build_status(plan_id=None, state_db=db, out=out)
    assert rc == 0
    body = out.getvalue()
    # Most-recent first ordering: bbb's row inserted second has higher
    # applied_at, so it's selected.
    assert "bbb" in body
    assert "completed" in body


def test_status_explicit_plan_id_renders_assignment_rollup(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    state.record_build_plan(
        plan_id="abc", applied_by="cli", plan_json="{}",
    )
    state.record_assignment(BuildAssignmentRecord(
        plan_id="abc", node_id="n1", image_ref="x:1",
        benchmark="b", status="done",
    ))
    state.record_assignment(BuildAssignmentRecord(
        plan_id="abc", node_id="n1", image_ref="x:2",
        benchmark="b", status="failed", error="oops",
    ))
    state.close()
    out = io.StringIO()
    rc = cmd_build_status(plan_id="abc", state_db=db, out=out)
    assert rc == 0
    body = out.getvalue()
    assert "abc" in body
    assert "done: 1" in body
    assert "failed: 1" in body
    assert "n1/x:2: oops" in body


def test_status_renders_registered_and_evicted_rollup(tmp_path: Path) -> None:
    """P1.6.g step 5 (#79): ``xrlenv build status`` should surface the
    ``registered`` (deferred) and ``evicted`` (cache reclaim) states
    so operators see why a ``completed`` plan still has unbuilt
    images. The status output labels them with the human-readable
    explanation, not the raw enum value alone."""
    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    state.record_build_plan(
        plan_id="xyz", applied_by="cli", plan_json="{}",
    )
    state.record_assignment(BuildAssignmentRecord(
        plan_id="xyz", node_id="n1", image_ref="x:1",
        benchmark="b", status="done",
    ))
    state.record_assignment(BuildAssignmentRecord(
        plan_id="xyz", node_id="n1", image_ref="x:huge",
        benchmark="b", status="registered",
    ))
    state.record_assignment(BuildAssignmentRecord(
        plan_id="xyz", node_id="n2", image_ref="x:old",
        benchmark="b", status="evicted",
    ))
    state.close()
    out = io.StringIO()
    rc = cmd_build_status(plan_id="xyz", state_db=db, out=out)
    assert rc == 0
    body = out.getvalue()
    assert "registered (deferred" in body
    assert "evicted (cache reclaim" in body
    assert "1\n" in body  # 1 of each


def test_status_renders_cancelled_rollup(tmp_path: Path) -> None:
    """Audit fix (post-2ebdaab): ``cancelled`` rows must appear in the
    assignment rollup so operators see them in the same place as
    ``done`` / ``failed`` rather than silently disappearing into the
    "X total" line.
    """
    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    state.record_build_plan(
        plan_id="cancel-rollup-aa", applied_by="cli", plan_json="{}",
    )
    state.update_build_plan_status("cancel-rollup-aa", "cancelled")
    state.record_assignment(BuildAssignmentRecord(
        plan_id="cancel-rollup-aa", node_id="n1", image_ref="x:1",
        benchmark="b", status="done",
    ))
    state.record_assignment(BuildAssignmentRecord(
        plan_id="cancel-rollup-aa", node_id="n1", image_ref="x:2",
        benchmark="b", status="cancelled", error="cancelled by operator",
    ))
    state.record_assignment(BuildAssignmentRecord(
        plan_id="cancel-rollup-aa", node_id="n2", image_ref="x:3",
        benchmark="b", status="cancelled", error="cancelled by operator",
    ))
    state.close()

    out = io.StringIO()
    rc = cmd_build_status(plan_id="cancel-rollup-aa", state_db=db, out=out)
    assert rc == 0
    body = out.getvalue()
    assert "assignments: 3 total" in body
    assert "cancelled (operator-cancelled" in body
    # Both cancelled rows counted (n=2).
    assert ": 2\n" in body


def test_status_unknown_plan_id_returns_nonzero(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    SqliteStateStore(db).close()
    out = io.StringIO()
    rc = cmd_build_status(plan_id="no-such", state_db=db, out=out)
    assert rc == 1
    assert "not found" in out.getvalue()


def test_status_accepts_unique_plan_id_prefix(tmp_path: Path) -> None:
    """Operators copy the 12-char short id out of the admin /builds
    panel; the CLI accepts any unique prefix (>=4 chars) so they
    don't have to paste the full SHA-256."""
    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    full_id = "89df31b1c020fb0ef0714fed21f2dbbf102822f03cba99de003033b137f653f2"
    state.record_build_plan(
        plan_id=full_id, applied_by="cli", plan_json="{}",
        name="test-plan",
    )
    state.update_build_plan_status(full_id, "completed")
    state.close()
    out = io.StringIO()
    # The 12-char admin-panel-style prefix.
    rc = cmd_build_status(plan_id=full_id[:12], state_db=db, out=out)
    assert rc == 0
    body = out.getvalue()
    assert full_id in body
    assert "completed" in body


def test_status_rejects_ambiguous_plan_id_prefix(tmp_path: Path) -> None:
    """When a prefix matches multiple plans, the CLI errors and
    lists candidates so the operator can disambiguate."""
    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    state.record_build_plan(
        plan_id="abcd111111111", applied_by="cli", plan_json="{}",
        name="alpha",
    )
    state.record_build_plan(
        plan_id="abcd222222222", applied_by="cli", plan_json="{}",
        name="beta",
    )
    state.close()
    out = io.StringIO()
    # 4-char prefix "abcd" matches both — past the too-short guard
    # but ambiguous against the populated table.
    rc = cmd_build_status(plan_id="abcd", state_db=db, out=out)
    assert rc == 1
    body = out.getvalue()
    assert "ambiguous" in body
    # Both candidates listed.
    assert "abcd111111111" in body
    assert "abcd222222222" in body


def test_status_rejects_too_short_prefix(tmp_path: Path) -> None:
    """Prefixes <4 chars are rejected up front (don't bother
    listing thousands of plans on a populated cluster)."""
    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    state.record_build_plan(
        plan_id="abc111", applied_by="cli", plan_json="{}",
    )
    state.close()
    out = io.StringIO()
    rc = cmd_build_status(plan_id="ab", state_db=db, out=out)
    assert rc == 1
    assert "too short" in out.getvalue()


def test_cancel_marks_plan_cancelled(tmp_path: Path) -> None:
    """`xrlenv build cancel` updates plan status to `cancelled` so
    operators can clear an in-flight plan without dropping to sqlite3."""
    from xrlenv.cli.commands import cmd_build_cancel

    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    state.record_build_plan(
        plan_id="stuck123abcdef", applied_by="cli", plan_json="{}",
        name="stuck-plan",
    )
    # record_build_plan inserts as in_flight.
    state.close()

    out = io.StringIO()
    rc = cmd_build_cancel(
        plan_id="stuck123abcdef", state_db=db, out=out,
    )
    assert rc == 0
    body = out.getvalue()
    assert "in_flight → cancelled" in body
    # Status persisted.
    state2 = SqliteStateStore(db)
    rec = state2.get_build_plan("stuck123abcdef")
    state2.close()
    assert rec is not None
    assert rec.status == "cancelled"


def test_cancel_accepts_plan_id_prefix(tmp_path: Path) -> None:
    """Cancel takes the same prefix shape as status."""
    from xrlenv.cli.commands import cmd_build_cancel

    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    state.record_build_plan(
        plan_id="cancel-by-prefix-id", applied_by="cli", plan_json="{}",
    )
    state.close()
    out = io.StringIO()
    rc = cmd_build_cancel(plan_id="cancel-by", state_db=db, out=out)
    assert rc == 0
    state2 = SqliteStateStore(db)
    rec = state2.get_build_plan("cancel-by-prefix-id")
    state2.close()
    assert rec is not None
    assert rec.status == "cancelled"


def test_cancel_already_terminal_is_noop(tmp_path: Path) -> None:
    """Cancel on a `completed` or already-`cancelled` plan is a no-op
    (returns 0 without re-updating). Idempotent."""
    from xrlenv.cli.commands import cmd_build_cancel

    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    state.record_build_plan(
        plan_id="done-plan-id", applied_by="cli", plan_json="{}",
    )
    state.update_build_plan_status("done-plan-id", "completed")
    state.close()

    out = io.StringIO()
    rc = cmd_build_cancel(plan_id="done-plan-id", state_db=db, out=out)
    assert rc == 0
    body = out.getvalue()
    assert "already terminal" in body
    # Status preserved.
    state2 = SqliteStateStore(db)
    rec = state2.get_build_plan("done-plan-id")
    state2.close()
    assert rec is not None
    assert rec.status == "completed"


def test_cancel_unknown_plan_id_returns_nonzero(tmp_path: Path) -> None:
    from xrlenv.cli.commands import cmd_build_cancel

    db = tmp_path / "state.db"
    SqliteStateStore(db).close()
    out = io.StringIO()
    rc = cmd_build_cancel(plan_id="no-such-plan", state_db=db, out=out)
    assert rc == 1
    assert "not found" in out.getvalue()


def test_status_missing_state_db_returns_2(tmp_path: Path) -> None:
    out = io.StringIO()
    rc = cmd_build_status(
        plan_id=None, state_db=tmp_path / "nope.db", out=out,
    )
    assert rc == 2
    assert "not found" in out.getvalue()


def test_apply_connect_host_dry_run_against_admin(tmp_path: Path) -> None:
    """P1.6.f cluster-RPC: with --connect-host, the CLI POSTs to the
    admin API (NOT LocalRuntime). Use the FastAPI TestClient via a
    custom httpx transport so the admin server runs in-process."""
    import json as _json

    import httpx
    from fastapi.testclient import TestClient
    from xrlenv.admin.server import AdminServerConfig, build_admin_app
    from xrlenv.cli import commands as _cli

    class _FakeCoordinator:
        async def apply(self, plan, **kw):
            from xrlenv.control.build_coordinator import BuildOutcome
            from xrlenv.control.image_planner import (
                PlacementResult,
                PlanAssignment,
            )

            assignments = (
                PlanAssignment(
                    image_ref="x:1", node_id="n1",  # type: ignore[arg-type]
                    benchmark="b", size_bytes=1024,
                ),
            )
            return BuildOutcome(
                plan_id="abcd1234deadbeef", status="dry_run",
                placement=PlacementResult(
                    assignments=assignments,
                    assignments_by_node={"n1": assignments},  # type: ignore[dict-item]
                ),
            )

    cfg = AdminServerConfig(
        state_db=tmp_path / "state.db",
        runs_root=tmp_path / "runs",
        port=0,
        build_coordinator=_FakeCoordinator(),
    )
    app = build_admin_app(cfg)
    test_client = TestClient(app)

    # Monkey-patch httpx.Client used inside _build_apply_via_admin to
    # route through TestClient. We do this by patching httpx at the
    # level the CLI imports it.

    class _RoutingClient:
        def __init__(self, *, base_url, headers, timeout):
            self._tc = test_client
            self._headers = headers

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def post(self, url, json):
            r = self._tc.post(url, json=json, headers=self._headers)
            return _Resp(r)

        def get(self, url):
            r = self._tc.get(url, headers=self._headers)
            return _Resp(r)

    class _Resp:
        def __init__(self, r) -> None:
            self.status_code = r.status_code
            self.text = r.text
            self._body = r.content

        def json(self):
            return _json.loads(self._body)

    out = io.StringIO()
    # The function imports httpx locally, so patch the module the
    # call resolves through.
    import sys as _sys

    fake_httpx = type("_F", (), {"Client": _RoutingClient, "HTTPError": httpx.HTTPError})
    _sys.modules["httpx"] = fake_httpx  # type: ignore[assignment]
    try:
        rc = _cli.cmd_build_apply(
            plan_path=None, benchmark="b",
            smoke=True, instances=None, all_=False,
            build_path=None, replication=None,
            reserved_runtime_gb=30, buffer_gb=10,
            dry_run=True, force=False,
            state_db=tmp_path / "state.db",
            runs_root=tmp_path / "runs",
            connect_host="example.test", connect_port=8080,
            operator_token=None,
            out=out,
        )
    finally:
        _sys.modules["httpx"] = httpx  # type: ignore[assignment]

    assert rc == 0
    body = out.getvalue()
    assert "abcd1234deadbeef" in body
    assert "dry_run" in body
    assert "x:1" in body


def test_apply_connect_host_threads_eager_to_admin(tmp_path: Path) -> None:
    """Audit P1.6.g-M1 fix: --eager on the cluster (--connect-host)
    path must reach the admin coordinator, not silently degrade to
    opportunistic mode. Asserts the CLI POSTs ``eager=True`` in the
    body when --eager is set."""
    import json as _json

    import httpx
    from fastapi.testclient import TestClient
    from xrlenv.admin.server import AdminServerConfig, build_admin_app
    from xrlenv.cli import commands as _cli

    captured_kw: dict[str, Any] = {}

    class _FakeCoordinator:
        async def apply(self, plan, **kw):  # type: ignore[no-untyped-def]
            captured_kw.update(kw)
            from xrlenv.control.build_coordinator import BuildOutcome
            from xrlenv.control.image_planner import PlacementResult

            return BuildOutcome(
                plan_id="abcd1234", status="dry_run",
                placement=PlacementResult(
                    assignments=(), assignments_by_node={},
                ),
            )

    cfg = AdminServerConfig(
        state_db=tmp_path / "state.db",
        runs_root=tmp_path / "runs",
        port=0,
        build_coordinator=_FakeCoordinator(),
    )
    test_client = TestClient(build_admin_app(cfg))

    captured_body: dict[str, Any] = {}

    class _RoutingClient:
        def __init__(self, *, base_url, headers, timeout):
            self._tc = test_client
            self._headers = headers

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def post(self, url, json):
            captured_body.update(json)
            r = self._tc.post(url, json=json, headers=self._headers)
            return _Resp(r)

        def get(self, url):
            r = self._tc.get(url, headers=self._headers)
            return _Resp(r)

    class _Resp:
        def __init__(self, r) -> None:
            self.status_code = r.status_code
            self.text = r.text
            self._body = r.content

        def json(self):
            return _json.loads(self._body)

    out = io.StringIO()
    import sys as _sys

    fake_httpx = type("_F", (), {
        "Client": _RoutingClient, "HTTPError": httpx.HTTPError,
    })
    _sys.modules["httpx"] = fake_httpx  # type: ignore[assignment]
    try:
        rc = _cli.cmd_build_apply(
            plan_path=None, benchmark="b",
            smoke=True, instances=None, all_=False,
            build_path=None, replication=None,
            reserved_runtime_gb=30, buffer_gb=10,
            dry_run=True, force=False, eager=True,
            state_db=tmp_path / "state.db",
            runs_root=tmp_path / "runs",
            connect_host="example.test", connect_port=8080,
            operator_token=None,
            out=out,
        )
    finally:
        _sys.modules["httpx"] = httpx  # type: ignore[assignment]

    assert rc == 0
    # Body posted to admin carries eager=True ...
    assert captured_body.get("eager") is True
    # ... and reaches the coordinator.apply() call with eager=True.
    assert captured_kw.get("eager") is True


def test_apply_connect_host_handles_unreachable_admin(tmp_path: Path) -> None:
    """When the admin URL is unreachable, the CLI exits 2 with a
    helpful error pointer at xrlenv up + --admin-port."""
    out = io.StringIO()
    rc = cmd_build_apply(
        plan_path=None, benchmark="terminal-bench-2",
        smoke=True, instances=None, all_=False,
        build_path=None, replication=None,
        reserved_runtime_gb=30, buffer_gb=10,
        dry_run=False, force=False,
        state_db=tmp_path / "state.db",
        runs_root=tmp_path / "runs",
        # Port 1 is privileged + unbindable in a normal user shell, so
        # connecting to it fails fast.
        connect_host="127.0.0.1", connect_port=1,
        operator_token=None,
        out=out,
    )
    assert rc == 2
    body = out.getvalue()
    assert "cannot reach admin" in body
    assert "xrlenv up" in body


def test_apply_refuses_when_live_cluster_active(tmp_path: Path) -> None:
    """Audit P1.6-H1 fix: ``xrlenv build apply`` is local-only today;
    running it against a state.db that's currently fronting a live
    control plane (heartbeats < 30s old) would corrupt the live
    registry. The CLI must refuse with a useful error."""
    import time

    from xrlenv.cli.commands import cmd_build_apply

    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    state.record_node_connected(
        "live-vm", backends=["docker"], stream_epoch="ep-1",
        instance_id="inst-1",
    )
    # Force a fresh heartbeat so the 30s staleness window doesn't
    # exclude the row.
    state.update_node_seen("live-vm", time.time())
    state.close()

    out = io.StringIO()
    rc = cmd_build_apply(
        plan_path=None, benchmark="terminal-bench-2",
        smoke=True, instances=None, all_=False,
        build_path=None, replication=None,
        reserved_runtime_gb=30, buffer_gb=10,
        dry_run=False, force=False,
        state_db=db, runs_root=tmp_path / "runs",
        out=out,
    )
    assert rc == 2
    body = out.getvalue()
    assert "refusing to run" in body
    assert "live-vm" in body


def test_cancel_connect_host_dispatches_via_admin(tmp_path: Path) -> None:
    """``xrlenv build cancel --plan ID --connect-host HOST`` POSTs to
    /api/build/cancel and prints the per-(node, image) summary the
    admin returns. End-to-end through the in-process FastAPI
    TestClient — same pattern as the apply --connect-host tests."""
    import json as _json

    import httpx
    from fastapi.testclient import TestClient
    from xrlenv.admin.server import AdminServerConfig, build_admin_app
    from xrlenv.cli import commands as _cli
    from xrlenv.control.state import (
        BuildAssignmentRecord,
        SqliteStateStore,
    )

    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    state.record_build_plan(
        plan_id="cancel-via-cluster-eeff", applied_by="op",
        plan_json="{}", name="cluster-cancel",
    )
    state.update_build_plan_status("cancel-via-cluster-eeff", "in_flight")
    state.record_assignment(BuildAssignmentRecord(
        plan_id="cancel-via-cluster-eeff", node_id="n1",
        image_ref="x:1", benchmark="b", status="building",
    ))
    state.close()

    fake_calls: list[str] = []

    class _FakeTransport:
        async def cancel_build_image(
            self, *, image_ref: str, timeout_s: float = 30.0,
        ) -> tuple[str, str]:
            fake_calls.append(image_ref)
            return ("ok", "")

    def _node_lookup(node_id: str):
        return _FakeTransport() if node_id == "n1" else None

    cfg = AdminServerConfig(
        state_db=db, runs_root=tmp_path / "runs", port=0,
        node_lookup=_node_lookup,
    )
    app = build_admin_app(cfg)
    test_client = TestClient(app)

    class _RoutingClient:
        def __init__(self, *, base_url, headers, timeout):
            self._tc = test_client
            self._headers = headers

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def post(self, url, json):
            r = self._tc.post(url, json=json, headers=self._headers)
            return _Resp(r)

    class _Resp:
        def __init__(self, r) -> None:
            self.status_code = r.status_code
            self.text = r.text
            self._body = r.content

        def json(self):
            return _json.loads(self._body)

    out = io.StringIO()
    import sys as _sys

    fake_httpx = type(
        "_F", (), {"Client": _RoutingClient, "HTTPError": httpx.HTTPError},
    )
    _sys.modules["httpx"] = fake_httpx  # type: ignore[assignment]
    try:
        rc = _cli.cmd_build_cancel(
            plan_id="cancel-via-cluster-eeff",
            state_db=db, out=out,
            connect_host="example.test", connect_port=8080,
            operator_token=None,
        )
    finally:
        _sys.modules["httpx"] = httpx  # type: ignore[assignment]

    assert rc == 0
    body = out.getvalue()
    assert "cancel-via-cluster-eeff" in body
    assert "1 assignment(s) cancelled" in body
    # The wire-level fake transport was actually invoked through the
    # admin's orchestrator.
    assert fake_calls == ["x:1"]
    # State.db reflects the cluster-side update.
    state2 = SqliteStateStore(db)
    try:
        plan = state2.get_build_plan("cancel-via-cluster-eeff")
        assert plan is not None
        assert plan.status == "cancelled"
    finally:
        state2.close()


def test_calibrate_writes_yaml_with_cluster_reported_sizes(tmp_path: Path) -> None:
    """End-to-end through the in-process FastAPI TestClient: feed
    a plan YAML to ``xrlenv build calibrate``, run the admin
    aggregation against fake report_images, write the calibrated
    YAML out, verify the size_hint_bytes / size_hint_source flipped."""
    import json as _json

    import httpx
    import yaml as _yaml
    from fastapi.testclient import TestClient
    from xrlenv.admin.server import AdminServerConfig, build_admin_app
    from xrlenv.cli import commands as _cli
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    plan_yaml = (
        "version: 1\n"
        "entries:\n"
        "  - image_ref: my/a:1\n"
        "    context_source: { type: registry }\n"
        "    placement:\n"
        "      size_hint_bytes: 999999\n"
        "      size_hint_source: heuristic\n"
        "  - image_ref: my/never-built:1\n"
        "    context_source: { type: registry }\n"
        "    placement:\n"
        "      size_hint_bytes: 999999\n"
        "      size_hint_source: heuristic\n"
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(plan_yaml)
    output_path = tmp_path / "plan.calibrated.yaml"

    state = SqliteStateStore(tmp_path / "state.db")
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    class _Transport:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(
                images=[
                    ImageStateRecord(
                        name="my/a:1", tier="cold",
                        size_bytes=2_500_000, in_use_count=0,
                        last_used_at=None, pinned=False,
                    ),
                ],
            )

    cfg = AdminServerConfig(
        state_db=tmp_path / "state.db",
        runs_root=tmp_path / "runs",
        port=0,
        node_lookup=lambda nid: _Transport() if nid == "n1" else None,
    )
    app = build_admin_app(cfg)
    test_client = TestClient(app)

    class _RoutingClient:
        def __init__(self, *, base_url, headers, timeout):
            self._tc = test_client
            self._headers = headers

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def post(self, url, json):
            r = self._tc.post(url, json=json, headers=self._headers)
            return _Resp(r)

    class _Resp:
        def __init__(self, r) -> None:
            self.status_code = r.status_code
            self.text = r.text
            self._body = r.content

        def json(self):
            return _json.loads(self._body)

    out = io.StringIO()
    import sys as _sys

    fake_httpx = type("_F", (), {"Client": _RoutingClient, "HTTPError": httpx.HTTPError})
    _sys.modules["httpx"] = fake_httpx  # type: ignore[assignment]
    try:
        rc = _cli.cmd_build_calibrate(
            plan_path=plan_path, output_path=output_path,
            out=out, connect_host="example.test", connect_port=8080,
            operator_token=None,
        )
    finally:
        _sys.modules["httpx"] = httpx  # type: ignore[assignment]

    assert rc == 0
    body = out.getvalue()
    assert "1 measured" in body
    assert "1 unmeasured" in body
    assert output_path.is_file()

    raw = _yaml.safe_load(output_path.read_text())
    entries_by_ref = {e["image_ref"]: e for e in raw["entries"]}
    assert entries_by_ref["my/a:1"]["placement"]["size_hint_bytes"] == 2_500_000
    assert entries_by_ref["my/a:1"]["placement"]["size_hint_source"] == "cluster-reported"
    # Unmeasured kept the operator-supplied hint.
    assert entries_by_ref["my/never-built:1"]["placement"]["size_hint_bytes"] == 999999
    assert entries_by_ref["my/never-built:1"]["placement"]["size_hint_source"] == "heuristic"

    # Plan-id-change warning landed (since at least one entry's
    # size_hint changed, the canonical plan body differs → fresh
    # plan_id). Operators get this nudge at the moment they run
    # calibrate so the new build_plans row later doesn't surprise
    # them.
    assert "fresh plan_id" in body
    assert "build_plans row" in body


def test_build_apply_tarball_max_bytes_threads_to_resolver(tmp_path: Path) -> None:
    """Audit response: ``xrlenv build apply --build-tarball-max-bytes``
    overrides ``DEFAULT_BUILD_TARBALL_MAX_BYTES`` end-to-end. Set the
    flag below the default and write a tarball above the override
    (but below the default) — the apply should reject with the
    operator-side cap, not silently accept against the bigger
    default."""
    from xrlenv.cli.commands import cmd_build_apply

    plan_yaml = (
        "version: 1\n"
        "entries:\n"
        "  - image_ref: huge-tar/img:1\n"
        "    context_source:\n"
        "      type: tarball\n"
        "      path: ctx.tar\n"
        "      dockerfile: Dockerfile\n"
        "    placement:\n"
        "      size_hint_bytes: 1024\n"
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(plan_yaml)
    (tmp_path / "ctx.tar").write_bytes(b"X" * (3 * 1024 * 1024))  # 3 MB

    out = io.StringIO()
    rc = cmd_build_apply(
        plan_path=plan_path,
        benchmark=None, smoke=False, instances=None, all_=False,
        build_path=None, replication=None,
        reserved_runtime_gb=30, buffer_gb=10,
        tarball_max_bytes=1 * 1024 * 1024,  # 1 MB cap; ctx is 3 MB
        dry_run=True, force=False,
        state_db=tmp_path / "state.db",
        runs_root=tmp_path / "runs",
        out=out,
    )
    assert rc == 2
    body = out.getvalue()
    assert "huge-tar/img:1" in body
    assert "over the" in body
    # Helps the operator find the right knob.
    assert "--build-tarball-max-bytes" in body


def test_calibrate_requires_connect_host(tmp_path: Path) -> None:
    """Without --connect-host there's nothing to measure; the
    CLI rejects with a clear error rather than silently no-op."""
    from xrlenv.cli.commands import cmd_build_calibrate

    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("version: 1\nentries: []\n")
    out = io.StringIO()
    rc = cmd_build_calibrate(
        plan_path=plan_path, output_path=tmp_path / "out.yaml",
        out=out, connect_host=None,
    )
    assert rc == 2
    assert "--connect-host is required" in out.getvalue()


def test_cancel_local_only_warns_about_in_flight(tmp_path: Path) -> None:
    """Without --connect-host, the cancel updates state.db only and
    the operator-facing message names that limitation explicitly so
    they don't think running cluster builds were interrupted."""
    from xrlenv.cli.commands import cmd_build_cancel

    db = tmp_path / "state.db"
    state = SqliteStateStore(db)
    state.record_build_plan(
        plan_id="local-only-cancel-ggjj", applied_by="op", plan_json="{}",
    )
    state.update_build_plan_status("local-only-cancel-ggjj", "in_flight")
    state.close()

    out = io.StringIO()
    rc = cmd_build_cancel(
        plan_id="local-only-cancel-ggjj", state_db=db, out=out,
    )
    assert rc == 0
    body = out.getvalue()
    assert "local-only cancel" in body
    assert "--connect-host" in body
