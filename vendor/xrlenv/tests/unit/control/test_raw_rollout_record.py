"""P1.7.B.3 — RawRolloutRecord + StateStore CRUD + contextvar + coordinator
lifecycle persistence + drop-in label merge.

Pins the case-2/3 evaluation tracking flow:

- StateStore (InMemory + Sqlite) round-trips RawRolloutRecord.
- Status enum: acquiring | running | released | cancelled | failed.
- ``rollout_metadata(...)`` context manager sets / restores the contextvar.
- Drop-in's create_container override merges metadata into outgoing labels.
- Coordinator writes ``acquiring`` at start, transitions to ``running`` on
  successful acquire, ``failed`` on acquire error, ``released`` on destroy.
- ``rollout_id != container_id`` — separate columns, separate identities.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from xrlenv.compat.metadata import (
    LABEL_ARTIFACT_PATH,
    LABEL_DISPLAYED_NAME,
    LABEL_GROUP_ID,
    current_rollout_metadata,
    metadata_to_labels,
    rollout_metadata,
)
from xrlenv.control.raw_container_service import RawContainerCoordinator
from xrlenv.control.state import (
    InMemoryStateStore,
    RawRolloutRecord,
    SqliteStateStore,
)
from xrlenv.errors import XRLEnvError

# ──────────────────────────────────────────────────────────────────────────────
# Fakes (mirror the scheduler / node shape this slice needs)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeQueryReply:
    present: bool


@dataclass
class _FakeRecord:
    rollout_id: str
    container_id: str
    container_name: str
    image: str


@dataclass
class _FakeNode:
    node_id: str = "node-A"
    backends: list[str] = field(default_factory=lambda: ["docker"])
    has_image: bool = True
    raise_on_acquire: Exception | None = None
    captured_labels: dict[str, str] | None = None

    def supported_backends(self) -> list[str]:
        return list(self.backends)

    async def query_image(self, image: str) -> _FakeQueryReply:
        return _FakeQueryReply(present=self.has_image)

    async def acquire_container(self, **kwargs: Any) -> _FakeRecord:
        if self.raise_on_acquire is not None:
            raise self.raise_on_acquire
        # Capture labels so tests can assert label flow.
        self.captured_labels = kwargs.get("labels")
        return _FakeRecord(
            rollout_id=kwargs["rollout_id"],
            container_id="c-001",
            container_name="cname-001",
            image=kwargs["image"],
        )

    async def destroy_container(self, **_kwargs: Any) -> None:
        pass


@dataclass
class _FakePlacement:
    node: Any
    backend: str = "docker"
    score: float = 1.0
    reservation_id: str = "fake-res-0"


@dataclass
class _FakeScheduler:
    """P1.7.A leak-fix: ``RawContainerCoordinator.acquire`` now calls
    ``commit_placement`` / ``release_placement`` to keep the real
    scheduler's ``_pending`` dict from accumulating phantom load. The
    fake records both so any test asserting the call pattern can
    introspect, and so the production code's calls don't AttributeError
    on a fake that pre-dates the lifecycle.
    """

    nodes: list[_FakeNode]
    image_aware_placement: bool = True
    commit_calls: list[Any] = field(default_factory=list)
    release_calls: list[Any] = field(default_factory=list)
    _next_reservation: int = 0

    def place(self, *_args: Any, **_kwargs: Any) -> _FakePlacement:
        reservation_id = f"fake-res-{self._next_reservation}"
        self._next_reservation += 1
        for node in self.nodes:
            if "docker" in node.supported_backends():
                return _FakePlacement(
                    node=node, reservation_id=reservation_id,
                )
        raise XRLEnvError("no docker-capable node")

    def commit_placement(self, placement: Any) -> None:
        self.commit_calls.append(placement.reservation_id)

    def release_placement(self, placement: Any) -> None:
        self.release_calls.append(placement.reservation_id)


# ──────────────────────────────────────────────────────────────────────────────
# StateStore CRUD (InMemory)
# ──────────────────────────────────────────────────────────────────────────────


def test_in_memory_record_and_get_round_trips() -> None:
    store = InMemoryStateStore()
    record = RawRolloutRecord(
        rollout_id="r-1",
        status="acquiring",
        image="busybox:1",
        artifact_path="/tmp/foo",
        displayed_name="instance-A",
        created_at=time.time(),
    )
    store.record_raw_rollout(record)

    fetched = store.get_raw_rollout("r-1")
    assert fetched == record
    # Missing rollout returns None (not KeyError).
    assert store.get_raw_rollout("r-missing") is None


def test_in_memory_double_record_raises() -> None:
    store = InMemoryStateStore()
    rec = RawRolloutRecord(
        rollout_id="r-1", status="acquiring", image="busybox:1",
        created_at=time.time(),
    )
    store.record_raw_rollout(rec)
    with pytest.raises(KeyError, match="already exists"):
        store.record_raw_rollout(rec)


def test_in_memory_status_transitions() -> None:
    store = InMemoryStateStore()
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-1", status="acquiring", image="busybox:1",
        created_at=time.time(),
    ))
    # acquiring → running
    store.update_raw_rollout(
        "r-1", status="running",
        node_id="node-A", container_id="c-001", container_name="cname-001",
    )
    rec = store.get_raw_rollout("r-1")
    assert rec is not None
    assert rec.status == "running"
    assert rec.container_id == "c-001"
    # running → released
    finish_ts = time.time()
    store.update_raw_rollout(
        "r-1", status="released", finished_at=finish_ts,
    )
    rec = store.get_raw_rollout("r-1")
    assert rec is not None
    assert rec.status == "released"
    assert rec.finished_at == finish_ts


def test_in_memory_list_filters_by_status() -> None:
    store = InMemoryStateStore()
    for i, status in enumerate(["acquiring", "running", "released", "failed"]):
        store.record_raw_rollout(RawRolloutRecord(
            rollout_id=f"r-{i}", status=status, image="busybox:1",  # type: ignore[arg-type]
            created_at=time.time() + i,  # newer = higher created_at
        ))
    all_rows = store.list_raw_rollouts()
    assert len(all_rows) == 4
    # Newest first.
    assert all_rows[0].rollout_id == "r-3"

    only_running = store.list_raw_rollouts(status="running")
    assert len(only_running) == 1
    assert only_running[0].rollout_id == "r-1"

    limited = store.list_raw_rollouts(limit=2)
    assert len(limited) == 2


# ──────────────────────────────────────────────────────────────────────────────
# StateStore CRUD (Sqlite)
# ──────────────────────────────────────────────────────────────────────────────


def test_sqlite_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    store = SqliteStateStore(db)
    rec = RawRolloutRecord(
        rollout_id="r-1", status="acquiring", image="swebench/sweb:latest",
        artifact_path="/repo/tmp/job-1/logs/...",
        displayed_name="astropy__astropy-7166",
        task_key="astropy__astropy-7166",
        group_id="run-2026-05-12-1530",
        created_at=time.time(),
    )
    store.record_raw_rollout(rec)
    fetched = store.get_raw_rollout("r-1")
    assert fetched == rec
    assert fetched is not None
    assert fetched.task_key == "astropy__astropy-7166"
    assert fetched.group_id == "run-2026-05-12-1530"


def test_sqlite_round_trip_no_task_key_or_group_id(tmp_path: Path) -> None:
    """Backwards compat: a record without task_key / group_id still
    serialises + deserialises cleanly. Pre-existing harness paths
    that don't pass either field keep working."""
    db = tmp_path / "state.db"
    store = SqliteStateStore(db)
    rec = RawRolloutRecord(
        rollout_id="r-1", status="acquiring", image="busybox:1",
        created_at=time.time(),
    )
    store.record_raw_rollout(rec)
    fetched = store.get_raw_rollout("r-1")
    assert fetched == rec
    assert fetched is not None
    assert fetched.task_key is None
    assert fetched.group_id is None


def test_sqlite_update_task_key_and_group_id(tmp_path: Path) -> None:
    """``update_raw_rollout`` accepts the new fields and persists
    them. Lets the coordinator backfill the keys onto an existing
    row when the labels arrive later than the initial record."""
    db = tmp_path / "state.db"
    store = SqliteStateStore(db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-1", status="acquiring", image="busybox:1",
        created_at=time.time(),
    ))
    store.update_raw_rollout(
        "r-1", task_key="bench/A", group_id="run-1",
    )
    rec = store.get_raw_rollout("r-1")
    assert rec is not None
    assert rec.task_key == "bench/A"
    assert rec.group_id == "run-1"


def _seed_store(store: Any) -> None:
    """Seed three rollouts spanning different task_key / group_id
    combos so filter tests have a non-trivial dataset."""
    now = time.time()
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-a", status="released", image="busybox:1",
        task_key="bench/instance-A", group_id="run-X",
        created_at=now,
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-b", status="released", image="busybox:1",
        task_key="bench/instance-B", group_id="run-X",
        created_at=now,
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-c", status="failed", image="busybox:1",
        task_key="bench/instance-A", group_id="run-Y",
        created_at=now,
    ))


@pytest.mark.parametrize(
    "store_factory",
    [
        lambda tmp_path: InMemoryStateStore(),
        lambda tmp_path: SqliteStateStore(tmp_path / "filter.db"),
    ],
    ids=["in_memory", "sqlite"],
)
def test_list_raw_rollouts_filter_by_task_key(
    store_factory: Any, tmp_path: Path,
) -> None:
    """Filtering by ``task_key`` returns only rollouts whose
    persisted task_key matches. Empty/None means no filter."""
    store = store_factory(tmp_path)
    _seed_store(store)
    matched = store.list_raw_rollouts(task_key="bench/instance-A")
    assert {r.rollout_id for r in matched} == {"r-a", "r-c"}
    no_match = store.list_raw_rollouts(task_key="bench/instance-Z")
    assert no_match == []
    # No filter = all rows.
    all_rows = store.list_raw_rollouts()
    assert {r.rollout_id for r in all_rows} == {"r-a", "r-b", "r-c"}


@pytest.mark.parametrize(
    "store_factory",
    [
        lambda tmp_path: InMemoryStateStore(),
        lambda tmp_path: SqliteStateStore(tmp_path / "filter.db"),
    ],
    ids=["in_memory", "sqlite"],
)
def test_list_raw_rollouts_filter_by_group_id(
    store_factory: Any, tmp_path: Path,
) -> None:
    """Filtering by ``group_id`` (operator-supplied via
    ``xrlenv.group_id`` label) returns just that group's rollouts.
    Backs the admin's "show me this run's tasks" workflow."""
    store = store_factory(tmp_path)
    _seed_store(store)
    matched = store.list_raw_rollouts(group_id="run-X")
    assert {r.rollout_id for r in matched} == {"r-a", "r-b"}


@pytest.mark.parametrize(
    "store_factory",
    [
        lambda tmp_path: InMemoryStateStore(),
        lambda tmp_path: SqliteStateStore(tmp_path / "filter.db"),
    ],
    ids=["in_memory", "sqlite"],
)
def test_list_raw_rollouts_combined_filters(
    store_factory: Any, tmp_path: Path,
) -> None:
    """status + task_key + group_id compose with AND semantics."""
    store = store_factory(tmp_path)
    _seed_store(store)
    matched = store.list_raw_rollouts(
        task_key="bench/instance-A", group_id="run-X", status="released",
    )
    assert {r.rollout_id for r in matched} == {"r-a"}


@pytest.mark.parametrize(
    "store_factory",
    [
        lambda tmp_path: InMemoryStateStore(),
        lambda tmp_path: SqliteStateStore(tmp_path / "filter.db"),
    ],
    ids=["in_memory", "sqlite"],
)
def test_count_raw_rollouts_honors_new_filters(
    store_factory: Any, tmp_path: Path,
) -> None:
    """``count_raw_rollouts`` mirrors ``list_raw_rollouts``'s filter
    set so admin's pagination total matches the rendered page."""
    store = store_factory(tmp_path)
    _seed_store(store)
    assert store.count_raw_rollouts(task_key="bench/instance-A") == 2
    assert store.count_raw_rollouts(group_id="run-X") == 2
    assert store.count_raw_rollouts(
        task_key="bench/instance-A", group_id="run-X",
    ) == 1
    assert store.count_raw_rollouts() == 3


def test_sqlite_update_field_validation(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    store = SqliteStateStore(db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-1", status="acquiring", image="busybox:1",
        created_at=time.time(),
    ))
    # Unknown field → ValueError
    with pytest.raises(ValueError, match="unknown fields"):
        store.update_raw_rollout("r-1", not_a_real_field="x")  # type: ignore[arg-type]
    # Unknown rollout → KeyError
    with pytest.raises(KeyError):
        store.update_raw_rollout("r-missing", status="released")


def test_sqlite_persists_across_reopen(tmp_path: Path) -> None:
    """Restart-survival check: a rollout recorded on one connection
    is queryable on a fresh connection (the slice's whole point —
    'all past runs are never tracked' is solved by writing through)."""
    db = tmp_path / "state.db"
    store1 = SqliteStateStore(db)
    rec = RawRolloutRecord(
        rollout_id="r-1", status="released",
        image="busybox:1",
        node_id="node-A", container_id="c-001",
        container_name="cname-001",
        displayed_name="instance-A",
        created_at=time.time(),
        finished_at=time.time() + 10,
    )
    store1.record_raw_rollout(rec)
    # New process / fresh connection.
    store2 = SqliteStateStore(db)
    fetched = store2.get_raw_rollout("r-1")
    assert fetched == rec


# ──────────────────────────────────────────────────────────────────────────────
# rollout_metadata contextvar
# ──────────────────────────────────────────────────────────────────────────────


def test_rollout_metadata_default_is_empty() -> None:
    meta = current_rollout_metadata()
    assert meta.artifact_path is None
    assert meta.displayed_name is None
    assert metadata_to_labels(meta) == {}


def test_rollout_metadata_sets_and_restores() -> None:
    with rollout_metadata(
        artifact_path="/tmp/foo", displayed_name="instance-A",
    ):
        meta = current_rollout_metadata()
        assert meta.artifact_path == "/tmp/foo"
        assert meta.displayed_name == "instance-A"
        labels = metadata_to_labels(meta)
        assert labels[LABEL_ARTIFACT_PATH] == "/tmp/foo"
        assert labels[LABEL_DISPLAYED_NAME] == "instance-A"
    # Restored on exit.
    assert current_rollout_metadata().artifact_path is None


def test_rollout_metadata_only_one_field_set() -> None:
    """Either field can be set independently; the other stays None +
    its label is omitted from the labels dict (no stray empty values)."""
    with rollout_metadata(displayed_name="instance-only"):
        labels = metadata_to_labels(current_rollout_metadata())
        assert labels == {LABEL_DISPLAYED_NAME: "instance-only"}
        assert LABEL_ARTIFACT_PATH not in labels


def test_rollout_metadata_group_id_emits_cancel_cohort_label() -> None:
    """group_id rides out as the bare ``xrlenv.group_id`` cancel-cohort label (not the
    ``xrlenv.rollout.*`` namespace) so every acquire in the block is tagged for
    ``terminate_raw_group`` — even harness acquires (harbor/swebench) that never pass labels."""
    with rollout_metadata(group_id="run-42"):
        meta = current_rollout_metadata()
        assert meta.group_id == "run-42"
        labels = metadata_to_labels(meta)
        assert labels == {LABEL_GROUP_ID: "run-42"}   # only the group label, bare key
    assert current_rollout_metadata().group_id is None  # restored on exit
    assert LABEL_GROUP_ID not in metadata_to_labels(current_rollout_metadata())


def test_rollout_metadata_unknown_kwarg_raises() -> None:
    """Typed-kwargs API: unknown keys land as TypeError via Python's
    normal kwargs validation. No silent drops."""
    with pytest.raises(TypeError), rollout_metadata(  # type: ignore[call-arg]
        artifact_path="/tmp", garbage="x",
    ):
        pass


def test_rollout_metadata_in_worker_thread() -> None:
    """The smoke driver's actual pattern: set the contextvar
    INSIDE the worker function (not in the submitter). The
    ``with rollout_metadata(...)`` block lives within
    ``_run_one_instance``, runs in the worker thread, and the
    drop-in's ``create_container`` (also called from the worker
    thread) reads it. No cross-thread propagation needed.

    Each worker thread has its own contextvar state — concurrent
    workers running with different metadata don't interfere.
    """
    captured: dict[str, dict[str, str | None]] = {}

    def _worker(name: str, path: str) -> None:
        with rollout_metadata(artifact_path=path, displayed_name=name):
            meta = current_rollout_metadata()
            captured[name] = {
                "artifact_path": meta.artifact_path,
                "displayed_name": meta.displayed_name,
            }

    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, path in [
            ("A", "/tmp/A"), ("B", "/tmp/B"),
            ("C", "/tmp/C"), ("D", "/tmp/D"),
        ]:
            pool.submit(_worker, name, path).result()

    assert captured["A"]["artifact_path"] == "/tmp/A"
    assert captured["A"]["displayed_name"] == "A"
    assert captured["D"]["artifact_path"] == "/tmp/D"
    # After all workers exit, no metadata leaks into the submitter.
    assert current_rollout_metadata().artifact_path is None


def test_rollout_metadata_isolated_across_concurrent_threads() -> None:
    """Each thread should see its own scoped metadata (contextvar
    copy-per-task semantics)."""
    barrier = threading.Barrier(2)
    results: dict[str, str] = {}

    def _worker(name: str, path: str) -> None:
        with rollout_metadata(artifact_path=path, displayed_name=name):
            barrier.wait()  # ensure both threads inside the block
            meta = current_rollout_metadata()
            results[name] = meta.artifact_path or ""

    t1 = threading.Thread(target=_worker, args=("A", "/tmp/A"))
    t2 = threading.Thread(target=_worker, args=("B", "/tmp/B"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["A"] == "/tmp/A"
    assert results["B"] == "/tmp/B"


# ──────────────────────────────────────────────────────────────────────────────
# Coordinator lifecycle persistence
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_coordinator_writes_record_on_acquire() -> None:
    state = InMemoryStateStore()
    node = _FakeNode()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]), state=state,
    )

    # Pass labels that the drop-in would have injected from a
    # ``with xrlenv.rollout_metadata(...)`` scope.
    session = await coord.acquire(
        image="busybox:1",
        labels={
            LABEL_ARTIFACT_PATH: "/tmp/run-1/instance-A",
            LABEL_DISPLAYED_NAME: "instance-A",
        },
    )

    rec = state.get_raw_rollout(session.rollout_id)
    assert rec is not None
    assert rec.status == "running"
    assert rec.image == "busybox:1"
    assert rec.node_id == "node-A"
    assert rec.container_id == "c-001"
    assert rec.container_name == "cname-001"
    assert rec.artifact_path == "/tmp/run-1/instance-A"
    assert rec.displayed_name == "instance-A"
    # ``rollout_id != container_id`` — separate identities.
    assert rec.rollout_id != rec.container_id


@pytest.mark.asyncio
async def test_coordinator_records_failed_on_acquire_error() -> None:
    state = InMemoryStateStore()
    node = _FakeNode(raise_on_acquire=RuntimeError("simulated wire error"))
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]), state=state,
    )

    with pytest.raises(RuntimeError, match="simulated wire error"):
        await coord.acquire(image="busybox:1")

    rows = state.list_raw_rollouts()
    assert len(rows) == 1
    rec = rows[0]
    assert rec.status == "failed"
    assert rec.error is not None
    assert "simulated wire error" in rec.error
    assert rec.finished_at is not None


@pytest.mark.asyncio
async def test_coordinator_records_released_on_destroy() -> None:
    state = InMemoryStateStore()
    node = _FakeNode()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]), state=state,
    )
    session = await coord.acquire(image="busybox:1")

    await coord.destroy(
        rollout_id=session.rollout_id,
        container_id=session.container_id,
    )

    rec = state.get_raw_rollout(session.rollout_id)
    assert rec is not None
    assert rec.status == "released"
    assert rec.finished_at is not None
    # Records ordered: created_at < finished_at.
    assert rec.finished_at >= rec.created_at


@pytest.mark.asyncio
async def test_coordinator_no_state_is_safe() -> None:
    """When the coordinator has no StateStore wired (older test
    fixtures, lighter-weight setups), acquire still works — record
    persistence silently skipped."""
    node = _FakeNode()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]),
        state=None,  # no StateStore
    )

    session = await coord.acquire(image="busybox:1")
    assert session.container_id == "c-001"
    # Destroy works too.
    await coord.destroy(
        rollout_id=session.rollout_id,
        container_id=session.container_id,
    )


@pytest.mark.asyncio
async def test_coordinator_passes_labels_to_node() -> None:
    """The labels dict (which carries the
    ``xrlenv.rollout.*`` keys from the drop-in's metadata merge)
    flows from the coordinator to the node transport — so docker
    sees the labels too, not just the cluster's record."""
    state = InMemoryStateStore()
    node = _FakeNode()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]), state=state,
    )

    await coord.acquire(
        image="busybox:1",
        labels={LABEL_DISPLAYED_NAME: "instance-A", "user.foo": "bar"},
    )

    # Node-side captured labels include both the xrlenv-reserved
    # key AND the operator's other labels — the coordinator
    # doesn't strip anything.
    assert node.captured_labels is not None
    assert node.captured_labels[LABEL_DISPLAYED_NAME] == "instance-A"
    assert node.captured_labels["user.foo"] == "bar"
