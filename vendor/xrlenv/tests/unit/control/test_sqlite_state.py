"""Tests for the StateStore Protocol — SqliteStateStore + InMemory parity.

Both stores must agree on every public surface (spec 20). The parametrized
fixture lets each test exercise both backends in one pass; sqlite-only tests
(persistence across reopen, WAL mode, schema integrity) live below.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from xrlenv.control.state import (
    InMemoryStateStore,
    PendingRolloutRecord,
    RawRolloutRecord,
    RolloutRecord,
    SandboxRecord,
    SqliteStateStore,
    StateStore,
)
from xrlenv.types import RolloutStatus, Step

# ──────────────────────────────────────────────────────────────────────────────
# Parametrized fixture: every test runs against InMemory + Sqlite
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(params=["inmem", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StateStore]:
    if request.param == "inmem":
        s: StateStore = InMemoryStateStore()
        yield s
    else:
        s = SqliteStateStore(tmp_path / "state.db")
        try:
            yield s
        finally:
            s.close()  # type: ignore[attr-defined]


def _rollout(rollout_id: str = "r1", **overrides: Any) -> RolloutRecord:
    base: dict[str, Any] = {
        "rollout_id": rollout_id,
        "template": "hello-shell",
        "status": RolloutStatus.STARTING,
    }
    base.update(overrides)
    return RolloutRecord(**base)


def _sandbox(sandbox_id: str = "sb1", **overrides: Any) -> SandboxRecord:
    base: dict[str, Any] = {
        "sandbox_id": sandbox_id,
        "backend": "docker",
        "backend_ref": "docker-cid",
        "stub_endpoint": "tcp://127.0.0.1:50000",
        "template": "hello-shell",
        "node_id": "local-laptop",
    }
    base.update(overrides)
    return SandboxRecord(**base)


def _step(index: int = 0, reward: float = 0.0) -> Step:
    return Step(
        index=index,
        action={"cmd": "echo hi"},
        obs={"stdout": "hi\n"},
        reward=reward,
        done=False,
        truncated=False,
        info={"steps": index + 1},
        ts=float(index) * 0.1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Parity tests (both stores)
# ──────────────────────────────────────────────────────────────────────────────


def test_insert_get_round_trip(store: StateStore) -> None:
    store.insert_rollout(_rollout(task_key="t-42", group_id="g-7"))
    fetched = store.get_rollout("r1")
    assert fetched.template == "hello-shell"
    assert fetched.task_key == "t-42"
    assert fetched.group_id == "g-7"
    assert fetched.status == RolloutStatus.STARTING


def test_duplicate_insert_raises(store: StateStore) -> None:
    store.insert_rollout(_rollout())
    with pytest.raises(KeyError):
        store.insert_rollout(_rollout())


def test_unknown_rollout_raises(store: StateStore) -> None:
    with pytest.raises(KeyError):
        store.get_rollout("missing")


def test_update_rollout_preserves_unspecified_fields(store: StateStore) -> None:
    store.insert_rollout(_rollout(task_key="orig-task"))
    store.update_rollout("r1", status=RolloutStatus.RUNNING)
    fetched = store.get_rollout("r1")
    assert fetched.status == RolloutStatus.RUNNING
    assert fetched.task_key == "orig-task"


def test_update_unknown_rollout_raises(store: StateStore) -> None:
    with pytest.raises(KeyError):
        store.update_rollout("missing", status=RolloutStatus.FAILED)


def test_append_step_accumulates_reward(store: StateStore) -> None:
    store.insert_rollout(_rollout())
    store.append_step("r1", _step(0, reward=0.5))
    store.append_step("r1", _step(1, reward=0.25))
    fetched = store.get_rollout("r1")
    assert len(fetched.steps) == 2
    assert pytest.approx(fetched.final_reward) == 0.75


def test_seal_trajectory_returns_steps(store: StateStore) -> None:
    store.insert_rollout(_rollout())
    store.append_step("r1", _step(0, reward=1.0))
    store.update_rollout("r1", status=RolloutStatus.FINISHED)
    traj = store.seal_trajectory("r1")
    assert traj.status == RolloutStatus.FINISHED
    assert traj.final_reward == 1.0
    assert len(traj.steps) == 1


def test_seal_trajectory_surfaces_node_id_in_metadata(store: StateStore) -> None:
    """The Trajectory returned by ``seal_trajectory`` must expose
    ``node_id`` via ``metadata`` so smoke + diagnostic callers can show
    which VM ran each rollout. Regression for the Scenario-1 acceptance
    summary printing ``'node': None`` even though the rollout had a
    well-defined home node.
    """
    store.insert_rollout(_rollout(node_id="aws-i-0123", metadata={"foo": "bar"}))
    store.update_rollout("r1", status=RolloutStatus.FINISHED)
    traj = store.seal_trajectory("r1")
    assert traj.metadata.get("node_id") == "aws-i-0123"
    assert traj.metadata.get("foo") == "bar"


def test_sandbox_lifecycle(store: StateStore) -> None:
    store.insert_sandbox(_sandbox(image="xrlenv/hello-shell:0.1"))
    fetched = store.get_sandbox("sb1")
    assert fetched.status == "running"
    assert fetched.image == "xrlenv/hello-shell:0.1"
    store.update_sandbox("sb1", status="destroying")
    assert store.get_sandbox("sb1").status == "destroying"
    store.remove_sandbox("sb1")
    with pytest.raises(KeyError):
        store.get_sandbox("sb1")


def test_idempotency_round_trip(store: StateStore) -> None:
    store.record_idempotent("req-1", "rollout-A")
    assert store.lookup_idempotent("req-1") == "rollout-A"
    assert store.lookup_idempotent("req-2") is None
    # Re-recording the same key should overwrite, not raise.
    store.record_idempotent("req-1", "rollout-B")
    assert store.lookup_idempotent("req-1") == "rollout-B"


def test_event_seq_monotonic(store: StateStore) -> None:
    e1 = store.append_event("kind1", rollout_id="r1")
    e2 = store.append_event("kind2", rollout_id="r2")
    assert e2.seq > e1.seq
    seqs = [e.seq for e in store.events_since(0)]
    assert seqs == [e1.seq, e2.seq]
    assert [e.seq for e in store.events_since(e1.seq)] == [e2.seq]


def test_pending_queue_drain(store: StateStore) -> None:
    p1 = PendingRolloutRecord(pending_id="p1", template="t", init_params={})
    p2 = PendingRolloutRecord(
        pending_id="p2", template="t", init_params={}, queue_partition="urgent"
    )
    store.enqueue_pending(p1)
    store.enqueue_pending(p2)
    assert {p.pending_id for p in store.list_pending()} == {"p1", "p2"}
    assert [p.pending_id for p in store.list_pending(partition="urgent")] == ["p2"]
    store.remove_pending("p1")
    assert [p.pending_id for p in store.list_pending()] == ["p2"]


def test_pending_duplicate_raises(store: StateStore) -> None:
    store.enqueue_pending(
        PendingRolloutRecord(pending_id="p1", template="t", init_params={})
    )
    with pytest.raises(KeyError):
        store.enqueue_pending(
            PendingRolloutRecord(pending_id="p1", template="t", init_params={})
        )


def test_list_rollouts_returns_all(store: StateStore) -> None:
    store.insert_rollout(_rollout("a"))
    store.insert_rollout(_rollout("b"))
    ids = sorted(r.rollout_id for r in store.list_rollouts())
    assert ids == ["a", "b"]


# ──────────────────────────────────────────────────────────────────────────────
# Sqlite-only tests
# ──────────────────────────────────────────────────────────────────────────────


def test_sqlite_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    s1 = SqliteStateStore(db)
    s1.insert_rollout(_rollout(task_key="persisted"))
    s1.append_step("r1", _step(0, reward=0.7))
    s1.update_rollout("r1", status=RolloutStatus.RUNNING, sandbox_id="sb-x")
    s1.close()

    s2 = SqliteStateStore(db)
    fetched = s2.get_rollout("r1")
    assert fetched.task_key == "persisted"
    assert fetched.status == RolloutStatus.RUNNING
    assert fetched.sandbox_id == "sb-x"
    assert pytest.approx(fetched.final_reward) == 0.7
    assert len(fetched.steps) == 1
    s2.close()


def test_sqlite_uses_wal_mode(tmp_path: Path) -> None:
    s = SqliteStateStore(tmp_path / "state.db")
    try:
        cur = s._conn.execute("PRAGMA journal_mode")  # type: ignore[attr-defined]
        mode = cur.fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        s.close()


def test_sqlite_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "dirs"
    s = SqliteStateStore(nested / "state.db")
    try:
        assert nested.exists()
    finally:
        s.close()


def test_sqlite_event_payload_round_trips(tmp_path: Path) -> None:
    s = SqliteStateStore(tmp_path / "state.db")
    try:
        s.append_event("kind.x", rollout_id="r1", payload={"nested": {"k": [1, 2, 3]}})
        events = list(s.events_since(0))
        assert len(events) == 1
        assert events[0].payload == {"nested": {"k": [1, 2, 3]}}
    finally:
        s.close()


# ──────────────────────────────────────────────────────────────────────────────
# Slice-4 follow-up: nodes table mirrors NodeRegistry membership.
# ──────────────────────────────────────────────────────────────────────────────


def test_sqlite_record_node_connected_inserts_row(tmp_path: Path) -> None:
    s = SqliteStateStore(tmp_path / "state.db")
    try:
        rec = s.record_node_connected(
            "gcp-1", backends=["docker"],
            stream_epoch="ep1", instance_id="inst1",
        )
        assert rec.status == "connected"
        rows = s.list_nodes()
        assert [r.node_id for r in rows] == ["gcp-1"]
        assert rows[0].backends == ["docker"]
        assert rows[0].instance_id == "inst1"
    finally:
        s.close()


def test_sqlite_record_node_disconnected_marks_lost(tmp_path: Path) -> None:
    s = SqliteStateStore(tmp_path / "state.db")
    try:
        s.record_node_connected("aws-1", backends=["docker"])
        s.record_node_disconnected("aws-1")
        rows = s.list_nodes()
        assert len(rows) == 1
        assert rows[0].status == "lost"
        assert s.list_nodes(status="connected") == []
    finally:
        s.close()


# ── prune_lost_nodes (startup reconciliation against the roster) ──────────────


def test_prune_lost_nodes_reaps_unrostered_lost(store: StateStore) -> None:
    """Only ``lost`` rows absent from ``keep`` are reaped; a rostered ``lost``
    node and a ``connected`` node survive."""
    store.record_node_connected("gone", backends=["docker"])
    store.record_node_disconnected("gone")               # lost + unrostered
    store.record_node_connected("kept-lost", backends=["docker"])
    store.record_node_disconnected("kept-lost")          # lost but rostered
    store.record_node_connected("live", backends=["docker"])  # connected

    pruned = store.prune_lost_nodes(keep={"kept-lost", "live"})

    assert pruned == ["gone"]
    assert {r.node_id for r in store.list_nodes()} == {"kept-lost", "live"}


def test_prune_lost_nodes_never_reaps_connected(store: StateStore) -> None:
    """A ``connected`` node is kept even when absent from the roster — only
    ``lost`` rows are reapable."""
    store.record_node_connected("live-unrostered", backends=["docker"])
    assert store.prune_lost_nodes(keep=set()) == []
    assert [r.node_id for r in store.list_nodes()] == ["live-unrostered"]


def test_prune_lost_nodes_clears_satellite_rows(store: StateStore) -> None:
    """Reaping a node also drops its ``node_health`` + ``node_aimd_limit``
    rows (rollout history in ``raw_rollouts`` is deliberately untouched)."""
    store.record_node_connected("gone", backends=["docker"])
    store.update_node_health("gone", '{"cpu": 1}')
    store.update_node_aimd_limit("gone", 42)
    store.record_node_disconnected("gone")

    assert store.prune_lost_nodes(keep=set()) == ["gone"]
    assert "gone" not in store.list_node_health()
    assert "gone" not in store.list_node_aimd_limits()


def test_prune_lost_nodes_noop_when_all_rostered(store: StateStore) -> None:
    store.record_node_connected("live", backends=["docker"])
    store.record_node_connected("rostered-lost", backends=["docker"])
    store.record_node_disconnected("rostered-lost")
    assert store.prune_lost_nodes(keep={"live", "rostered-lost"}) == []
    assert len(store.list_nodes()) == 2


# ── prune_expired (spec 20 retention GC matrix) ───────────────────────────────


def _raw_for_prune(
    rollout_id: str, status: str, *, created_at: float, finished_at: float | None = None
) -> Any:
    from xrlenv.control.state import RawRolloutRecord

    return RawRolloutRecord(
        rollout_id=rollout_id, status=status, image="img:1",
        created_at=created_at, finished_at=finished_at,
    )


def test_prune_expired_audit_by_age(store: StateStore) -> None:
    store.append_audit("rpc", role="node")
    now = time.time()
    # Fresh row survives a 30d window evaluated at `now`.
    assert store.prune_expired(
        now=now, audit_retention_days=30,
        events_retention_days=None, raw_rollout_retention_days=None,
    )["audit"] == 0
    assert len(list(store.audit_since(0))) == 1
    # Evaluated 31 days later, the same row is now past the 30d window.
    counts = store.prune_expired(
        now=now + 31 * 86400, audit_retention_days=30,
        events_retention_days=None, raw_rollout_retention_days=None,
    )
    assert counts["audit"] == 1
    assert list(store.audit_since(0)) == []


def test_prune_expired_events_by_age(store: StateStore) -> None:
    store.append_event("sandbox_created", rollout_id="r1")
    now = time.time()
    assert store.prune_expired(
        now=now + 15 * 86400, audit_retention_days=None,
        events_retention_days=14, raw_rollout_retention_days=None,
    )["events"] == 1
    assert list(store.events_since(0)) == []


def test_prune_expired_raw_rollouts_terminal_only(store: StateStore) -> None:
    old = time.time() - 100 * 86400
    store.record_raw_rollout(
        _raw_for_prune("old-released", "released", created_at=old, finished_at=old)
    )
    store.record_raw_rollout(  # active + old — must NOT be pruned
        _raw_for_prune("old-running", "running", created_at=old)
    )
    store.record_raw_rollout(  # terminal but fresh — must NOT be pruned
        _raw_for_prune(
            "new-released", "released",
            created_at=time.time(), finished_at=time.time(),
        )
    )
    counts = store.prune_expired(
        now=time.time(), audit_retention_days=None,
        events_retention_days=None, raw_rollout_retention_days=14,
    )
    assert counts["raw_rollouts"] == 1
    assert store.get_raw_rollout("old-released") is None
    assert store.get_raw_rollout("old-running") is not None
    assert store.get_raw_rollout("new-released") is not None


def test_prune_expired_none_windows_skip_everything(store: StateStore) -> None:
    store.append_audit("rpc")
    counts = store.prune_expired(
        now=time.time() + 999 * 86400,
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=None,
    )
    assert counts == {"audit": 0, "events": 0, "raw_rollouts": 0}
    assert len(list(store.audit_since(0))) == 1


def test_sqlite_reconnect_preserves_connected_at(tmp_path: Path) -> None:
    """A node that drops + reconnects keeps its original ``connected_at``
    while it's continuously attached, but resets the timestamp once the
    registry has marked it ``lost`` between attachments."""
    s = SqliteStateStore(tmp_path / "state.db")
    try:
        first = s.record_node_connected("flap-1", backends=["docker"])
        # Same-stream re-register (no disconnect in between): same connected_at.
        second = s.record_node_connected("flap-1", backends=["docker"])
        assert second.connected_at == first.connected_at
        # Drop + reconnect: connected_at advances.
        s.record_node_disconnected("flap-1")
        third = s.record_node_connected("flap-1", backends=["docker"])
        assert third.connected_at > first.connected_at
    finally:
        s.close()


# ── P6 step-2c — node isolation capability + pinned-CPU observability ──────────


def test_sqlite_record_node_connected_persists_isolation_capable(tmp_path: Path) -> None:
    s = SqliteStateStore(tmp_path / "state.db")
    try:
        rec = s.record_node_connected("cap-1", backends=["docker"], isolation_capable=True)
        assert rec.isolation_capable is True
        rows = {r.node_id: r for r in s.list_nodes()}
        assert rows["cap-1"].isolation_capable is True
        # Default is False (a node that didn't advertise / pre-P6 agent).
        s.record_node_connected("plain-1", backends=["docker"])
        assert {r.node_id: r for r in s.list_nodes()}["plain-1"].isolation_capable is False
    finally:
        s.close()


def test_sqlite_update_node_pinned_cpus(tmp_path: Path) -> None:
    s = SqliteStateStore(tmp_path / "state.db")
    try:
        s.record_node_connected("cap-1", backends=["docker"])
        # Unknown until first heartbeat mirror.
        assert (s.list_nodes()[0].pinned_cpus_free, s.list_nodes()[0].pinned_cpus_total) == (0, 0)
        s.update_node_pinned_cpus("cap-1", free=6, total=8)
        row = s.list_nodes()[0]
        assert (row.pinned_cpus_free, row.pinned_cpus_total) == (6, 8)
        # No-op for an unknown node (UPDATE matches no row, doesn't raise).
        s.update_node_pinned_cpus("ghost", free=1, total=2)
        assert {r.node_id for r in s.list_nodes()} == {"cap-1"}
    finally:
        s.close()


def test_sqlite_nodes_migration_adds_isolation_columns(tmp_path: Path) -> None:
    """A dev database that predates the P6 columns migrates cleanly: the ALTERs
    add the columns with 0 defaults, existing rows read false/0/0, and writes
    against the migrated table work."""
    db = tmp_path / "old.db"
    con = sqlite3.connect(str(db))
    con.execute(
        """
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY, status TEXT NOT NULL,
            connected_at REAL NOT NULL, last_seen_at REAL NOT NULL,
            stream_epoch TEXT, instance_id TEXT,
            backends_json TEXT NOT NULL DEFAULT '[]'
        )
        """,
    )
    con.execute(
        "INSERT INTO nodes (node_id, status, connected_at, last_seen_at, backends_json) "
        "VALUES ('old-1', 'connected', 1.0, 2.0, '[\"docker\"]')",
    )
    con.commit()
    con.close()

    s = SqliteStateStore(db)  # __init__ runs _migrate → adds the P6 columns
    try:
        old = {r.node_id: r for r in s.list_nodes()}["old-1"]
        assert old.isolation_capable is False
        assert (old.pinned_cpus_free, old.pinned_cpus_total) == (0, 0)
        # Writes against the migrated table work.
        s.record_node_connected("new-1", backends=["docker"], isolation_capable=True)
        s.update_node_pinned_cpus("new-1", free=3, total=4)
        new = {r.node_id: r for r in s.list_nodes()}["new-1"]
        assert new.isolation_capable is True
        assert (new.pinned_cpus_free, new.pinned_cpus_total) == (3, 4)
    finally:
        s.close()


# ──────────────────────────────────────────────────────────────────────────────
# list_rollouts_page (demand-paged admin view)
# ──────────────────────────────────────────────────────────────────────────────


def test_list_rollouts_page_basic_pagination_newest_first(tmp_path: Path) -> None:
    """Returned records are newest-first; offset/limit slice correctly and
    the N+1 probe populates ``has_next`` accurately."""
    s = SqliteStateStore(tmp_path / "state.db")
    now = time.time()
    try:
        for i in range(5):
            s.insert_rollout(_rollout(f"r-{i}", created_at=now - float(i)))
        page1, has_next1 = s.list_rollouts_page(limit=2, offset=0)
        assert [r.rollout_id for r in page1] == ["r-0", "r-1"]
        assert has_next1 is True

        page2, has_next2 = s.list_rollouts_page(limit=2, offset=2)
        assert [r.rollout_id for r in page2] == ["r-2", "r-3"]
        assert has_next2 is True

        page3, has_next3 = s.list_rollouts_page(limit=2, offset=4)
        assert [r.rollout_id for r in page3] == ["r-4"]
        assert has_next3 is False
    finally:
        s.close()


def test_list_rollouts_page_filter_status_and_template(tmp_path: Path) -> None:
    """status and template WHERE clauses are independently and jointly
    applied so that only matching rows are returned."""
    s = SqliteStateStore(tmp_path / "state.db")
    now = time.time()
    try:
        s.insert_rollout(
            _rollout(
                "r-run-a", template="alpha",
                status=RolloutStatus.RUNNING, created_at=now,
            )
        )
        s.insert_rollout(
            _rollout(
                "r-fin-a", template="alpha",
                status=RolloutStatus.FINISHED, created_at=now - 1,
            )
        )
        s.insert_rollout(
            _rollout(
                "r-run-b", template="beta",
                status=RolloutStatus.RUNNING, created_at=now - 2,
            )
        )

        rows, _ = s.list_rollouts_page(status="running", limit=10, offset=0)
        assert {r.rollout_id for r in rows} == {"r-run-a", "r-run-b"}

        rows, _ = s.list_rollouts_page(template="alpha", limit=10, offset=0)
        assert {r.rollout_id for r in rows} == {"r-run-a", "r-fin-a"}

        rows, _ = s.list_rollouts_page(
            status="running", template="alpha", limit=10, offset=0,
        )
        assert [r.rollout_id for r in rows] == ["r-run-a"]
    finally:
        s.close()


def test_list_rollouts_page_created_after_filter(tmp_path: Path) -> None:
    """created_after excludes rollouts older than the cutoff, matching the
    ``since`` duration filter the admin view passes down."""
    s = SqliteStateStore(tmp_path / "state.db")
    now = time.time()
    try:
        s.insert_rollout(_rollout("r-new", created_at=now - 10.0))
        s.insert_rollout(_rollout("r-old", created_at=now - 3600.0))

        rows, has_next = s.list_rollouts_page(
            created_after=now - 60.0, limit=10, offset=0,
        )
        assert [r.rollout_id for r in rows] == ["r-new"]
        assert has_next is False
    finally:
        s.close()


def test_list_rollouts_page_invalid_args_raise(tmp_path: Path) -> None:
    """limit < 1 and offset < 0 are programmer errors — raise ValueError."""
    s = SqliteStateStore(tmp_path / "state.db")
    try:
        with pytest.raises(ValueError, match="limit"):
            s.list_rollouts_page(limit=0, offset=0)
        with pytest.raises(ValueError, match="offset"):
            s.list_rollouts_page(limit=1, offset=-1)
    finally:
        s.close()


# ──────────────────────────────────────────────────────────────────────────────
# Slice-9b additive migration coverage (D9 from
# notes/deferred_audit_todos.md): SqliteStateStore._migrate must run
# cleanly against a pre-9b database that lacks the
# ``effective_resources_json`` and ``image`` columns. The contract:
#
#   - opening such a db must not raise
#   - the columns must be present after open
#   - existing rows return ``None`` for the new columns
#   - new rows round-trip the values verbatim
# ──────────────────────────────────────────────────────────────────────────────


def test_sqlite_migration_adds_effective_resources_json_and_image(
    tmp_path: Path,
) -> None:
    import sqlite3

    db_path = tmp_path / "pre9b.db"

    # Pre-seed a database with a sandboxes table that PRE-DATES the
    # 9b column additions. We deliberately omit
    # ``effective_resources_json`` and ``image`` from the CREATE,
    # then insert a row to simulate a real dev database that
    # carried over from before the 9b ALTER TABLE.
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(
            """
            CREATE TABLE sandboxes (
                sandbox_id    TEXT PRIMARY KEY,
                backend       TEXT NOT NULL,
                backend_ref   TEXT NOT NULL,
                stub_endpoint TEXT NOT NULL,
                template      TEXT NOT NULL,
                node_id       TEXT NOT NULL,
                rollout_id    TEXT,
                status        TEXT NOT NULL DEFAULT 'running',
                owner_count   INTEGER NOT NULL DEFAULT 1,
                created_at    REAL NOT NULL
            );
            INSERT INTO sandboxes (
                sandbox_id, backend, backend_ref, stub_endpoint,
                template, node_id, status, owner_count, created_at
            ) VALUES (
                'sb-legacy', 'docker', 'cid-legacy', 'tcp://127.0.0.1:0',
                't', 'n-1', 'running', 1, 100.0
            );
            """,
        )
        raw.commit()
    finally:
        raw.close()

    # Open via SqliteStateStore. The migration must run additively.
    s = SqliteStateStore(db_path)
    try:
        # New columns are present.
        cols = {
            row["name"]
            for row in s._conn.execute(  # type: ignore[attr-defined]
                "PRAGMA table_info(sandboxes)",
            )
        }
        assert "effective_resources_json" in cols
        assert "image" in cols

        # Pre-existing row reads back with None on the new columns.
        legacy = s.get_sandbox("sb-legacy")
        assert legacy.effective_resources_json is None
        assert legacy.image is None

        # New rows round-trip the new columns verbatim.
        s.insert_sandbox(
            SandboxRecord(
                sandbox_id="sb-fresh",
                backend="docker",
                backend_ref="cid-fresh",
                stub_endpoint="tcp://127.0.0.1:0",
                template="t",
                image="im/t:1",
                node_id="n-1",
                effective_resources_json='{"cpu_limit": 2.0}',
            ),
        )
        fresh = s.get_sandbox("sb-fresh")
        assert fresh.image == "im/t:1"
        assert fresh.effective_resources_json == '{"cpu_limit": 2.0}'
    finally:
        s.close()


def test_sqlite_migration_is_idempotent_when_columns_already_present(
    tmp_path: Path,
) -> None:
    """Opening the same db twice must not error — the second open's
    migration sees the columns already exist and no-ops. Pin this so a
    refactor to ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` (which
    SQLite doesn't support) doesn't slip in unnoticed.
    """
    db_path = tmp_path / "twice.db"
    s1 = SqliteStateStore(db_path)
    s1.close()
    s2 = SqliteStateStore(db_path)  # must not raise
    try:
        cols = {
            row["name"]
            for row in s2._conn.execute(  # type: ignore[attr-defined]
                "PRAGMA table_info(sandboxes)",
            )
        }
        assert {"effective_resources_json", "image"}.issubset(cols)
    finally:
        s2.close()


def test_is_closed_flag_tracks_close(tmp_path: Path) -> None:
    """``is_closed`` is False on a live store and True after
    ``close()`` — the flag shutdown-race-sensitive callers
    (``NodeRegistry.deregister``) check before a mirror write."""
    s = SqliteStateStore(tmp_path / "closeflag.db")
    assert s.is_closed is False
    s.close()
    assert s.is_closed is True


# ──────────────────────────────────────────────────────────────────────────────
# Raw-rollout aggregates (admin /users + /nodes distribution figures)
# ──────────────────────────────────────────────────────────────────────────────


def _raw(rollout_id: str, owner: str, status: str, node: str | None) -> RawRolloutRecord:
    return RawRolloutRecord(
        rollout_id=rollout_id, status=status, image="img:1",
        owner_id=owner, node_id=node,
    )


def _seed_raw_mix(store: StateStore) -> None:
    for rec in (
        _raw("a1", "alice", "released", "node-A"),
        _raw("a2", "alice", "released", "node-A"),
        _raw("a3", "alice", "failed", "node-B"),
        _raw("a4", "alice", "running", "node-A"),
        _raw("b1", "bob", "released", "node-B"),
        _raw("b2", "bob", "reaped", "node-B"),
        _raw("b3", "bob", "cancelled", None),  # not yet assigned
    ):
        store.record_raw_rollout(rec)


def test_aggregate_raw_rollouts_by_owner_status(store: StateStore) -> None:
    _seed_raw_mix(store)
    agg = store.aggregate_raw_rollouts_by_owner_status()
    assert agg["alice"] == {"released": 2, "failed": 1, "running": 1}
    assert agg["bob"] == {"released": 1, "reaped": 1, "cancelled": 1}
    # Totals reconcile with a per-owner count.
    assert sum(agg["alice"].values()) == 4
    assert set(agg) == {"alice", "bob"}


def test_count_raw_rollouts_by_node(store: StateStore) -> None:
    _seed_raw_mix(store)
    by_node = store.count_raw_rollouts_by_node()
    assert by_node["node-A"] == 3
    assert by_node["node-B"] == 3
    # The not-yet-assigned rollout buckets under the None key.
    assert by_node[None] == 1
    assert sum(by_node.values()) == 7


def test_raw_aggregates_empty_store(store: StateStore) -> None:
    assert store.aggregate_raw_rollouts_by_owner_status() == {}
    assert store.count_raw_rollouts_by_node() == {}


# ──────────────────────────────────────────────────────────────────────────────
# aggregate_raw_rollouts_all_time_by_owner_status (CHANGE B / audit M1)
#
# Returns {owner: {status: count}} = live raw_rollouts PLUS the durable
# owner_rollout_lifetime tally, read as ONE consistent snapshot. Tests cover
# both InMemoryStateStore and SqliteStateStore (via the ``store`` fixture).
# ──────────────────────────────────────────────────────────────────────────────


def _raw_for_agg(
    rollout_id: str,
    owner: str,
    status: str,
    *,
    created_at: float | None = None,
    finished_at: float | None = None,
) -> RawRolloutRecord:
    """Convenience factory for aggregate tests.

    ``finished_at`` is set automatically for terminal statuses so
    ``prune_expired`` can sweep them (it checks ``finished_at`` for
    terminal rows). ``created_at`` defaults to a far-past timestamp
    so rows are expired on any 14-day retention window.
    """
    terminal = {"released", "cancelled", "failed", "reaped", "capacity_rejected"}
    ca = created_at if created_at is not None else (time.time() - 200 * 86400)
    if finished_at is None and status in terminal:
        finished_at = ca  # expired terminal row
    return RawRolloutRecord(
        rollout_id=rollout_id,
        status=status,  # type: ignore[arg-type]
        image="img:1",
        owner_id=owner,
        created_at=ca,
        finished_at=finished_at,
    )


def test_aggregate_all_time_empty_store_returns_empty(store: StateStore) -> None:
    """Empty store → aggregate_raw_rollouts_all_time_by_owner_status returns {}."""
    assert store.aggregate_raw_rollouts_all_time_by_owner_status() == {}


def test_aggregate_all_time_live_only_rows(store: StateStore) -> None:
    """When no prune has run (all rows still live), all-time == live aggregate."""
    store.record_raw_rollout(_raw_for_agg("a1", "alice", "running"))
    store.record_raw_rollout(_raw_for_agg("a2", "alice", "released"))
    store.record_raw_rollout(_raw_for_agg("b1", "bob", "failed"))

    agg = store.aggregate_raw_rollouts_all_time_by_owner_status()
    assert agg["alice"]["running"] == 1
    assert agg["alice"]["released"] == 1
    assert agg["bob"]["failed"] == 1
    assert set(agg) == {"alice", "bob"}


def test_aggregate_all_time_after_prune_equals_pre_prune_counts(
    store: StateStore,
) -> None:
    """The critical no-loss invariant across the GC boundary.

    Insert N rows, record the per-owner/status counts BEFORE prune, then
    prune (moves expired terminal rows to lifetime), then verify
    aggregate_raw_rollouts_all_time_by_owner_status() returns the same
    counts. Proves no row is lost or double-counted at the GC seam.
    """
    fresh_ts = time.time()
    # Old terminal rows — will be pruned into lifetime.
    store.record_raw_rollout(_raw_for_agg("a1", "alice", "released"))
    store.record_raw_rollout(_raw_for_agg("a2", "alice", "released"))
    store.record_raw_rollout(_raw_for_agg("a3", "alice", "failed"))
    store.record_raw_rollout(_raw_for_agg("b1", "bob", "cancelled"))
    # Fresh terminal row — stays live after prune.
    store.record_raw_rollout(
        _raw_for_agg(
            "a4", "alice", "released",
            created_at=fresh_ts, finished_at=fresh_ts,
        )
    )
    # Active row — never pruned.
    store.record_raw_rollout(_raw_for_agg("a5", "alice", "running"))

    # Record expected counts BEFORE any prune.
    expected_alice = {"released": 3, "failed": 1, "running": 1}
    expected_bob = {"cancelled": 1}

    # Prune: moves old terminal rows to lifetime.
    store.prune_expired(
        now=time.time(),
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=14,
    )

    # All-time aggregate must equal the pre-prune counts.
    agg = store.aggregate_raw_rollouts_all_time_by_owner_status()
    assert agg.get("alice") == expected_alice, (
        f"alice mismatch after prune: got {agg.get('alice')!r}, "
        f"expected {expected_alice!r}"
    )
    assert agg.get("bob") == expected_bob, (
        f"bob mismatch after prune: got {agg.get('bob')!r}, "
        f"expected {expected_bob!r}"
    )


def test_aggregate_all_time_multiple_owners_and_statuses_no_cross_contamination(
    store: StateStore,
) -> None:
    """Multiple owners and multiple statuses are bucketed independently."""
    # Alice has two statuses; Bob has two statuses. No overlap.
    store.record_raw_rollout(_raw_for_agg("a1", "alice", "released"))
    store.record_raw_rollout(_raw_for_agg("a2", "alice", "released"))
    store.record_raw_rollout(_raw_for_agg("a3", "alice", "reaped"))
    store.record_raw_rollout(_raw_for_agg("b1", "bob", "failed"))
    store.record_raw_rollout(_raw_for_agg("b2", "bob", "cancelled"))

    agg = store.aggregate_raw_rollouts_all_time_by_owner_status()

    assert agg["alice"] == {"released": 2, "reaped": 1}
    assert agg["bob"] == {"failed": 1, "cancelled": 1}
    # Cross-contamination guard: alice has no "failed", bob has no "released".
    assert "failed" not in agg["alice"]
    assert "released" not in agg["bob"]


def test_aggregate_all_time_lifetime_only_after_full_prune(
    store: StateStore,
) -> None:
    """After ALL rows are pruned into lifetime, all-time still returns them.

    This exercises the path where raw_rollouts is empty but
    owner_rollout_lifetime holds all the history.
    """
    store.record_raw_rollout(_raw_for_agg("r1", "carol", "released"))
    store.record_raw_rollout(_raw_for_agg("r2", "carol", "failed"))

    # Prune everything (both rows are old terminal).
    store.prune_expired(
        now=time.time(),
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=14,
    )

    # raw_rollouts is now empty.
    assert store.list_raw_rollouts() == []

    # But all-time aggregate must still return the lifetime tallies.
    agg = store.aggregate_raw_rollouts_all_time_by_owner_status()
    assert agg == {"carol": {"released": 1, "failed": 1}}
