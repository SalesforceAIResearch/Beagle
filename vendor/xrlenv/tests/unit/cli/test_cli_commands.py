"""Tests for the Slice 5b operator CLI command implementations.

Covers the read-only commands (nodes / rollouts / replay / events / tail /
attach). Each test seeds a temp ``state.db`` + ``runs_root`` so command
functions exercise the same path the dispatcher uses without spinning up
a real control plane.

The ``up`` subcommand is exercised separately in
``test_runtime_and_cli.py``-style integration tests if needed; we don't
unit-test it here because it'd require booting a gRPC server.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from xrlenv.backends.base import ResourceSpec
from xrlenv.cli.commands import (
    cmd_attach,
    cmd_audit,
    cmd_db_prune,
    cmd_db_vacuum,
    cmd_events,
    cmd_nodes,
    cmd_replay,
    cmd_rollouts,
    cmd_tail,
    parse_duration,
)
from xrlenv.control.state import (
    RawRolloutRecord,
    RolloutRecord,
    SandboxRecord,
    SqliteStateStore,
)
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateManifest,
)
from xrlenv.control.trajectory_sink import PlatformJsonlSink
from xrlenv.types import RolloutStatus, Step

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def state_db(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    return root


def _seed_rollout(
    store: SqliteStateStore,
    *,
    rollout_id: str,
    template: str = "obs-t",
    status: RolloutStatus = RolloutStatus.RUNNING,
    node_id: str = "node-A",
    sandbox_id: str | None = None,
    final_reward: float = 0.0,
    created_offset_s: float = 0.0,
) -> RolloutRecord:
    record = RolloutRecord(
        rollout_id=rollout_id,
        template=template,
        status=status,
        node_id=node_id,
        sandbox_id=sandbox_id,
        final_reward=final_reward,
        created_at=time.time() - created_offset_s,
        last_touched_at=time.time() - created_offset_s,
    )
    store.insert_rollout(record)
    return record


def _seed_sandbox(
    store: SqliteStateStore,
    *,
    sandbox_id: str,
    node_id: str = "node-A",
    rollout_id: str | None = None,
    template: str = "obs-t",
) -> None:
    store.insert_sandbox(
        SandboxRecord(
            sandbox_id=sandbox_id,
            backend="docker",
            backend_ref=f"cid-{sandbox_id}",
            stub_endpoint="tcp://127.0.0.1:0",
            template=template,
            node_id=node_id,
            rollout_id=rollout_id,
        )
    )


def _manifest(name: str = "obs-t") -> TemplateManifest:
    return TemplateManifest(
        name=name, version="0.1", digest=f"sha256:{name}",
        image=f"im/{name}:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# parse_duration
# ──────────────────────────────────────────────────────────────────────────────


def test_parse_duration_units() -> None:
    assert parse_duration("30s") == 30
    assert parse_duration("5m") == 300
    assert parse_duration("2h") == 7200
    assert parse_duration("1d") == 86400


def test_parse_duration_invalid() -> None:
    with pytest.raises(ValueError):
        parse_duration("forever")
    with pytest.raises(ValueError):
        parse_duration("5min")


# ──────────────────────────────────────────────────────────────────────────────
# nodes
# ──────────────────────────────────────────────────────────────────────────────


def test_cmd_nodes_text_output(state_db: Path, tmp_path: Path) -> None:
    store = SqliteStateStore(state_db)
    _seed_sandbox(store, sandbox_id="sb-1", node_id="node-A")
    _seed_sandbox(store, sandbox_id="sb-2", node_id="node-A")
    _seed_sandbox(store, sandbox_id="sb-3", node_id="node-B")
    store.close()

    nodes_yaml = tmp_path / "nodes.yaml"
    nodes_yaml.write_text(
        yaml.safe_dump(
            {
                "nodes": [
                    {"id": "node-A", "cloud": "gcp", "expected_address": "10.0.0.1"},
                    {"id": "node-C", "cloud": "aws", "expected_address": "10.0.0.5"},
                ]
            }
        )
    )

    out = io.StringIO()
    rc = cmd_nodes(state_db=state_db, nodes_yaml=nodes_yaml, output_format="text", out=out)
    assert rc == 0
    body = out.getvalue()
    # Both rostered nodes show up; node-B (active but not rostered) too.
    assert "node-A" in body and "node-B" in body and "node-C" in body
    # Active count for node-A is 2.
    assert "2" in body
    # New STATUS column lands. None of these nodes have a registry row,
    # so all should show ``absent`` status (rostered/active but not
    # currently attached via gRPC).
    assert "STATUS" in body
    assert "absent" in body


def test_cmd_nodes_shows_live_attached_nodes_via_state_mirror(
    state_db: Path, tmp_path: Path,
) -> None:
    """A node that's connected to the live registry but has zero active
    sandboxes still appears in ``xrlenv nodes`` (as ``connected``).

    Pre-Slice-4, the CLI hid these because it cross-referenced only
    ``nodes.yaml`` plus ``state.list_sandboxes()``. The
    ``state.record_node_connected()`` mirror written by
    :class:`NodeRegistry` now surfaces them directly.
    """
    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "aws-i-1", backends=["docker"],
        stream_epoch="ep1", instance_id="inst1",
    )
    store.close()

    nodes_yaml = tmp_path / "nodes.yaml"
    nodes_yaml.write_text(yaml.safe_dump({"nodes": []}))

    out = io.StringIO()
    rc = cmd_nodes(
        state_db=state_db, nodes_yaml=nodes_yaml,
        output_format="json", out=out,
    )
    assert rc == 0
    payload = json.loads(out.getvalue())
    by_id = {n["id"]: n for n in payload}
    assert "aws-i-1" in by_id
    assert by_id["aws-i-1"]["status"] == "connected"
    assert by_id["aws-i-1"]["active_sandboxes"] == 0
    assert by_id["aws-i-1"]["rostered"] is False


def test_cmd_nodes_shows_lost_status_after_disconnect(
    state_db: Path, tmp_path: Path,
) -> None:
    store = SqliteStateStore(state_db)
    store.record_node_connected("flap", backends=["docker"])
    store.record_node_disconnected("flap")
    store.close()

    nodes_yaml = tmp_path / "nodes.yaml"
    nodes_yaml.write_text(yaml.safe_dump({"nodes": []}))

    out = io.StringIO()
    rc = cmd_nodes(
        state_db=state_db, nodes_yaml=nodes_yaml,
        output_format="json", out=out,
    )
    assert rc == 0
    payload = json.loads(out.getvalue())
    by_id = {n["id"]: n for n in payload}
    assert by_id["flap"]["status"] == "lost"


def test_cmd_nodes_json_output(state_db: Path, tmp_path: Path) -> None:
    store = SqliteStateStore(state_db)
    _seed_sandbox(store, sandbox_id="sb-1", node_id="node-X")
    store.close()

    nodes_yaml = tmp_path / "nodes.yaml"
    nodes_yaml.write_text(yaml.safe_dump({"nodes": [{"id": "node-X", "cloud": "gcp"}]}))

    out = io.StringIO()
    rc = cmd_nodes(state_db=state_db, nodes_yaml=nodes_yaml, output_format="json", out=out)
    assert rc == 0
    payload = json.loads(out.getvalue())
    by_id = {n["id"]: n for n in payload}
    assert by_id["node-X"]["rostered"] is True
    assert by_id["node-X"]["active_sandboxes"] == 1


def test_cmd_nodes_missing_state_db_raises(tmp_path: Path) -> None:
    out = io.StringIO()
    with pytest.raises(FileNotFoundError, match=r"state\.db not found"):
        cmd_nodes(state_db=tmp_path / "missing.db", out=out)


def test_cmd_nodes_accepts_both_address_and_expected_address(
    state_db: Path, tmp_path: Path,
) -> None:
    """Regression for audit M1 against commit b7f3b50.

    The spec-09 example key is ``expected_address``; the typed loader at
    xrlenv/control/nodes_yaml.py uses ``address``. ``cmd_nodes`` must
    surface either rather than silently emitting ``null`` when only one
    is set.
    """
    SqliteStateStore(state_db).close()  # empty store

    nodes_yaml = tmp_path / "nodes.yaml"
    nodes_yaml.write_text(
        yaml.safe_dump(
            {
                "nodes": [
                    {"id": "node-legacy", "address": "10.0.0.1"},
                    {"id": "node-spec",   "expected_address": "10.0.0.2"},
                    # Both set: spec key wins (forward-compat).
                    {"id": "node-both",
                     "address": "10.0.0.3-OLD",
                     "expected_address": "10.0.0.3"},
                ]
            }
        )
    )

    out = io.StringIO()
    rc = cmd_nodes(
        state_db=state_db, nodes_yaml=nodes_yaml,
        output_format="json", out=out,
    )
    assert rc == 0
    payload = json.loads(out.getvalue())
    by_id = {n["id"]: n for n in payload}
    assert by_id["node-legacy"]["expected_address"] == "10.0.0.1"
    assert by_id["node-spec"]["expected_address"] == "10.0.0.2"
    assert by_id["node-both"]["expected_address"] == "10.0.0.3"


# ──────────────────────────────────────────────────────────────────────────────
# rollouts
# ──────────────────────────────────────────────────────────────────────────────


def test_cmd_rollouts_filters_status_template_since(state_db: Path) -> None:
    store = SqliteStateStore(state_db)
    _seed_rollout(store, rollout_id="rid-1", template="t-a", status=RolloutStatus.RUNNING)
    _seed_rollout(store, rollout_id="rid-2", template="t-a", status=RolloutStatus.FINISHED)
    _seed_rollout(store, rollout_id="rid-3", template="t-b", status=RolloutStatus.FINISHED)
    _seed_rollout(
        store, rollout_id="rid-old", template="t-a",
        status=RolloutStatus.FINISHED, created_offset_s=3600,
    )
    store.close()

    out = io.StringIO()
    rc = cmd_rollouts(
        state_db=state_db, status="finished", template="t-a", since="10m",
        output_format="json", out=out,
    )
    assert rc == 0
    payload = json.loads(out.getvalue())
    ids = {r["rollout_id"] for r in payload}
    # Only rid-2 matches all three filters.
    assert ids == {"rid-2"}


def test_cmd_rollouts_text_table(state_db: Path) -> None:
    store = SqliteStateStore(state_db)
    _seed_rollout(store, rollout_id="rid-A", template="hello", status=RolloutStatus.RUNNING)
    store.close()
    out = io.StringIO()
    rc = cmd_rollouts(state_db=state_db, output_format="text", out=out)
    assert rc == 0
    body = out.getvalue()
    assert "ROLLOUT_ID" in body and "STATUS" in body
    assert "rid-A" in body and "running" in body and "hello" in body


# ──────────────────────────────────────────────────────────────────────────────
# replay
# ──────────────────────────────────────────────────────────────────────────────


def _open_and_seal_run(
    runs_root: Path, *, rollout_id: str = "rid-replay", final_reward: float = 0.7
) -> None:
    sink = PlatformJsonlSink(runs_root)
    sink.open(rollout_id=rollout_id, manifest=_manifest(), init={}, node_id="nid")
    sink.record_step(rollout_id, Step(
        index=0, action={"a": 1}, obs={"o": 1}, reward=0.0, done=False,
        truncated=False, info={}, ts=0.0,
    ))
    sink.seal(
        rollout_id=rollout_id, status=RolloutStatus.FINISHED,
        reason=None, final_reward=final_reward, metadata={},
    )


def test_cmd_replay_text(runs_root: Path) -> None:
    _open_and_seal_run(runs_root)
    out = io.StringIO()
    rc = cmd_replay("rid-replay", runs_root=runs_root, output_format="text", out=out)
    assert rc == 0
    body = out.getvalue()
    assert "rid-replay" in body
    assert "obs-t" in body
    assert "finished" in body


def test_cmd_replay_json(runs_root: Path) -> None:
    _open_and_seal_run(runs_root, final_reward=0.42)
    out = io.StringIO()
    rc = cmd_replay("rid-replay", runs_root=runs_root, output_format="json", out=out)
    assert rc == 0
    payload = json.loads(out.getvalue())
    assert payload["rollout_id"] == "rid-replay"
    assert pytest.approx(payload["final_reward"]) == 0.42


def test_cmd_replay_missing_returns_error(runs_root: Path) -> None:
    out = io.StringIO()
    rc = cmd_replay("never-existed", runs_root=runs_root, out=out)
    assert rc == 1
    assert "no run dir" in out.getvalue() or "not found" in out.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# events
# ──────────────────────────────────────────────────────────────────────────────


def test_cmd_events_filters_by_rollout(state_db: Path) -> None:
    store = SqliteStateStore(state_db)
    store.append_event("rollout.start", rollout_id="rid-1", payload={"k": "v1"})
    store.append_event("rollout.start", rollout_id="rid-2", payload={"k": "v2"})
    store.append_event("rollout.finish", rollout_id="rid-1", payload={"k": "v3"})
    store.close()

    out = io.StringIO()
    rc = cmd_events(state_db=state_db, rollout_id="rid-1", output_format="json", out=out)
    assert rc == 0
    payload = json.loads(out.getvalue())
    assert {e["kind"] for e in payload} == {"rollout.start", "rollout.finish"}
    assert all(e["rollout_id"] == "rid-1" for e in payload)


def test_cmd_events_filters_by_since(state_db: Path) -> None:
    store = SqliteStateStore(state_db)
    # Append + then push the ts back so it's "old".
    rec = store.append_event("rollout.fail", rollout_id="rid-old", payload={})
    store._conn.execute(  # type: ignore[attr-defined]
        "UPDATE events SET ts = ? WHERE seq = ?",
        (time.time() - 7200, rec.seq),
    )
    store._conn.commit()  # type: ignore[attr-defined]
    store.append_event("rollout.start", rollout_id="rid-new", payload={})
    store.close()

    out = io.StringIO()
    rc = cmd_events(state_db=state_db, since="5m", output_format="json", out=out)
    assert rc == 0
    payload = json.loads(out.getvalue())
    ids = {e["rollout_id"] for e in payload}
    # Only the new event survives the 5-minute cutoff.
    assert ids == {"rid-new"}


# ──────────────────────────────────────────────────────────────────────────────
# audit (spec 19) — separate table from `events`; used to surface
# auth.token_used / auth.denied so the operator can verify that node
# bidi streams attached even before any rollouts have run.
# ──────────────────────────────────────────────────────────────────────────────


def test_cmd_audit_emits_all_rows_by_default(state_db: Path) -> None:
    store = SqliteStateStore(state_db)
    store.append_audit("auth.token_used", role="node", method="/x.NodeControl/Stream", source="10.0.0.1")
    store.append_audit("auth.denied", role=None, method="/x.NodeControl/Stream", result="denied")
    store.close()

    out = io.StringIO()
    rc = cmd_audit(state_db=state_db, output_format="json", out=out)
    assert rc == 0
    payload = json.loads(out.getvalue())
    kinds = {row["kind"] for row in payload}
    assert {"auth.token_used", "auth.denied"} <= kinds


def test_cmd_audit_filters_by_kind_and_role(state_db: Path) -> None:
    store = SqliteStateStore(state_db)
    store.append_audit("auth.token_used", role="node", method="/x.NodeControl/Stream")
    store.append_audit("auth.token_used", role="consumer", method="/x.RolloutSvc/Run")
    store.append_audit("auth.denied", role=None, method="/x.NodeControl/Stream", result="denied")
    store.close()

    out = io.StringIO()
    rc = cmd_audit(
        state_db=state_db, kind="auth.token_used", role="node",
        output_format="json", out=out,
    )
    assert rc == 0
    payload = json.loads(out.getvalue())
    assert len(payload) == 1
    assert payload[0]["kind"] == "auth.token_used"
    assert payload[0]["role"] == "node"


def test_cmd_audit_filters_by_since(state_db: Path) -> None:
    store = SqliteStateStore(state_db)
    rec = store.append_audit("auth.token_used", role="node")
    store._conn.execute(  # type: ignore[attr-defined]
        "UPDATE audit SET ts = ? WHERE seq = ?",
        (time.time() - 7200, rec.seq),
    )
    store._conn.commit()  # type: ignore[attr-defined]
    store.append_audit("auth.token_used", role="consumer")
    store.close()

    out = io.StringIO()
    rc = cmd_audit(state_db=state_db, since="5m", output_format="json", out=out)
    assert rc == 0
    payload = json.loads(out.getvalue())
    roles = {row["role"] for row in payload}
    assert roles == {"consumer"}


def test_cmd_audit_text_output_renders_table(state_db: Path) -> None:
    store = SqliteStateStore(state_db)
    store.append_audit("auth.token_used", role="node", method="/m", source="src")
    store.close()

    out = io.StringIO()
    rc = cmd_audit(state_db=state_db, output_format="text", out=out)
    assert rc == 0
    body = out.getvalue()
    assert "SEQ" in body and "KIND" in body and "ROLE" in body
    assert "auth.token_used" in body
    assert "node" in body


# ──────────────────────────────────────────────────────────────────────────────
# tail
# ──────────────────────────────────────────────────────────────────────────────


def test_cmd_tail_prints_existing_lines(runs_root: Path) -> None:
    sink = PlatformJsonlSink(runs_root)
    sink.open(rollout_id="rid-tail", manifest=_manifest(), init={}, node_id="nid")
    for idx in range(3):
        sink.record_step("rid-tail", Step(
            index=idx, action={}, obs={}, reward=0.0, done=False,
            truncated=False, info={}, ts=float(idx),
        ))
    out = io.StringIO()
    rc = cmd_tail("rid-tail", runs_root=runs_root, stop_after_s=0.05, out=out)
    assert rc == 0
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [p["index"] for p in parsed] == [0, 1, 2]


def test_cmd_tail_missing_run_dir_errors(runs_root: Path) -> None:
    out = io.StringIO()
    rc = cmd_tail("never-existed", runs_root=runs_root, stop_after_s=0.0, out=out)
    assert rc == 1
    assert "no run dir" in out.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# attach
# ──────────────────────────────────────────────────────────────────────────────


def test_cmd_attach_prints_snapshot_and_tails_log(
    state_db: Path, runs_root: Path,
) -> None:
    # Seed both state.db (snapshot data) and the on-disk run dir
    # (coordinator.log via PlatformJsonlSink.record_event).
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="rid-att", template="obs-t",
        status=RolloutStatus.RUNNING, sandbox_id="sb-att",
    )
    _seed_sandbox(store, sandbox_id="sb-att", rollout_id="rid-att")
    store.append_event("rollout.start", rollout_id="rid-att", payload={"foo": "bar"})
    store.close()

    sink = PlatformJsonlSink(runs_root)
    sink.open(rollout_id="rid-att", manifest=_manifest(), init={}, node_id="node-A")
    sink.record_event("rid-att", "rollout.start", {"foo": "bar"})
    sink.record_event("rid-att", "step.0", {"reward": 0.0})

    out = io.StringIO()
    rc = cmd_attach(
        "rid-att", state_db=state_db, runs_root=runs_root,
        stop_after_s=0.05, out=out,
    )
    assert rc == 0
    body = out.getvalue()
    assert "rid-att" in body
    assert "obs-t" in body
    assert "running" in body
    assert "sb-att" in body
    # Should have followed coordinator.log and printed at least the seed events.
    assert "rollout.start" in body
    assert "step.0" in body


def test_cmd_attach_unknown_rollout_returns_error(
    state_db: Path, runs_root: Path,
) -> None:
    SqliteStateStore(state_db).close()  # create empty DB
    out = io.StringIO()
    rc = cmd_attach("never-existed", state_db=state_db, runs_root=runs_root, out=out)
    assert rc == 1


def test_cmd_attach_does_not_flip_journal_mode(
    state_db: Path, runs_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase-0 ``attach`` is READ-ONLY inspection (no container, no interactive
    shell — just a snapshot + event tail), so it must open the state store
    read-only and never flip a live TRUNCATE prod DB to WAL. Regression guard
    for the H5 read-safety contract, parallel to cmd_build_status: this
    exercises the FULL snapshot read path (rollout + sandbox + events + log
    tail), not just the not-found early return.

    (The audit filed this as "needs a live container" — that's a misdiagnosis:
    phase-0 attach never attaches to a container, so its journal-preservation
    contract is fully unit-testable.)
    """
    def _write_version(db: Path) -> int:
        # SQLite header byte 18: 1 = rollback journal (TRUNCATE), 2 = WAL.
        with open(db, "rb") as f:
            return f.read(20)[18]

    # Seed a rollout into a TRUNCATE-mode DB (as the control plane runs it),
    # then clear the env so the attach open mimics a login-user invocation with
    # XRLENV_SQLITE_JOURNAL_MODE unset (the exact H2/H5 foot-gun scenario).
    monkeypatch.setenv("XRLENV_SQLITE_JOURNAL_MODE", "TRUNCATE")
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="rid-ro", template="obs-t",
        status=RolloutStatus.RUNNING, sandbox_id="sb-ro",
    )
    _seed_sandbox(store, sandbox_id="sb-ro", rollout_id="rid-ro")
    store.append_event("rollout.start", rollout_id="rid-ro", payload={"k": "v"})
    store.close()
    monkeypatch.delenv("XRLENV_SQLITE_JOURNAL_MODE", raising=False)

    assert _write_version(state_db) == 1, "precondition: TRUNCATE DB (write_version 1)"

    out = io.StringIO()
    rc = cmd_attach(
        "rid-ro", state_db=state_db, runs_root=runs_root,
        stop_after_s=0.05, out=out,
    )
    assert rc == 0
    assert "rid-ro" in out.getvalue()  # full read path actually ran
    assert _write_version(state_db) == 1, (
        "cmd_attach flipped the journal mode to WAL — phase-0 attach must open "
        "the store read-only (H5 read-safety); it must NOT run PRAGMA journal_mode"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher smoke (no subcommand → exit 2)
# ──────────────────────────────────────────────────────────────────────────────


def test_dispatcher_unknown_subcommand_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from xrlenv.cli.__main__ import main as cli_main

    with pytest.raises(SystemExit) as excinfo:
        cli_main(["nope"])
    assert excinfo.value.code != 0


def test_dispatcher_rollouts_routes_to_cmd_rollouts(
    state_db: Path, runs_root: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke-test the dispatcher actually invokes cmd_rollouts."""
    import xrlenv.cli.__main__ as cli_module

    captured: dict[str, Any] = {}

    def _fake_cmd_rollouts(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli_module, "cmd_rollouts", _fake_cmd_rollouts)
    rc = cli_module.main(
        [
            "--state-db", str(state_db),
            "--runs-root", str(runs_root),
            "rollouts", "--status", "running",
        ]
    )
    assert rc == 0
    assert captured["status"] == "running"
    assert captured["state_db"] == state_db


# ── db prune / vacuum (spec 20 retention GC) ──────────────────────────────────


def test_cmd_db_prune_removes_old_terminal_raw_rollouts(state_db: Path) -> None:
    store = SqliteStateStore(state_db)
    old = time.time() - 100 * 86400
    store.record_raw_rollout(
        RawRolloutRecord(
            rollout_id="old", status="released", image="i",
            created_at=old, finished_at=old,
        )
    )
    store.record_raw_rollout(  # active + old — must survive
        RawRolloutRecord(rollout_id="run", status="running", image="i", created_at=old)
    )
    store.close()

    out = io.StringIO()
    rc = cmd_db_prune(
        state_db=state_db,
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=1,
        out=out,
    )
    assert rc == 0
    assert "raw_rollouts=1" in out.getvalue()

    check = SqliteStateStore(state_db)
    try:
        assert check.get_raw_rollout("old") is None
        assert check.get_raw_rollout("run") is not None
    finally:
        check.close()


def test_cmd_db_vacuum_runs_and_reports(state_db: Path) -> None:
    SqliteStateStore(state_db).close()  # materialize the db file
    out = io.StringIO()
    rc = cmd_db_vacuum(state_db=state_db, out=out)
    assert rc == 0
    assert "VACUUM complete" in out.getvalue()


def test_cmd_db_vacuum_missing_db_raises(tmp_path: Path) -> None:
    out = io.StringIO()
    with pytest.raises(FileNotFoundError):
        cmd_db_vacuum(state_db=tmp_path / "nope.db", out=out)
