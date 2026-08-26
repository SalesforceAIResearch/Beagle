"""Unit tests for _overview_blocking nodes_lost / node_connected / node_rostered
(2026-08-21 CP-resilience additions).

Pins the new overview fields that surface node-liveness status so operators
see a nonzero nodes_lost whenever a rostered node is not currently connected.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from xrlenv.admin import AdminServerConfig, build_admin_app
from xrlenv.admin.server import _overview_blocking
from xrlenv.control.state import SqliteStateStore

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


def _cfg(
    state_db: Path,
    runs_root: Path,
    nodes_yaml: Path | None = None,
) -> AdminServerConfig:
    return AdminServerConfig(
        state_db=state_db,
        runs_root=runs_root,
        port=0,
        nodes_yaml=nodes_yaml,
    )


def _nodes_yaml(tmp_path: Path, node_ids: list[str]) -> Path:
    p = tmp_path / "nodes.yaml"
    p.write_text(yaml.safe_dump({
        "nodes": [
            {"id": nid, "cloud": "aws", "expected_address": f"10.0.0.{i}"}
            for i, nid in enumerate(node_ids, 1)
        ]
    }))
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Empty-DB early return
# ──────────────────────────────────────────────────────────────────────────────


def test_empty_db_early_return_has_nodes_lost_zero(
    state_db: Path, runs_root: Path,
) -> None:
    """When state.db doesn't exist yet the early-return dict must include
    nodes_lost: 0, node_connected: 0, node_rostered: 0."""
    cfg = _cfg(state_db, runs_root)
    assert not state_db.exists()
    result = _overview_blocking(cfg, time.time())
    assert result["nodes_lost"] == 0
    assert result["node_connected"] == 0
    assert result["node_rostered"] == 0
    assert result["state_db_present"] is False


def test_empty_db_early_return_via_http(
    state_db: Path, runs_root: Path,
) -> None:
    """The same zero values surface via the /api/overview JSON endpoint."""
    cfg = _cfg(state_db, runs_root)
    client = TestClient(build_admin_app(cfg))
    payload = client.get("/api/overview").json()
    assert payload["nodes_lost"] == 0
    assert payload["node_connected"] == 0
    assert payload["node_rostered"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# Fully-connected fleet: nodes_lost == 0
# ──────────────────────────────────────────────────────────────────────────────


def test_fully_connected_fleet_nodes_lost_is_zero(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """All rostered nodes are connected → nodes_lost == 0."""
    nodes = ["node-a", "node-b", "node-c"]
    ny = _nodes_yaml(tmp_path, nodes)

    store = SqliteStateStore(state_db)
    for nid in nodes:
        store.record_node_connected(
            node_id=nid,
            stream_epoch=f"ep-{nid}",
            instance_id=f"inst-{nid}",
            backends=["docker"],
        )
    store.close()

    cfg = _cfg(state_db, runs_root, nodes_yaml=ny)
    result = _overview_blocking(cfg, time.time())
    assert result["nodes_lost"] == 0
    assert result["node_connected"] == 3
    assert result["node_rostered"] == 3


def test_fully_connected_fleet_via_http(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    nodes = ["node-x", "node-y"]
    ny = _nodes_yaml(tmp_path, nodes)

    store = SqliteStateStore(state_db)
    for nid in nodes:
        store.record_node_connected(
            node_id=nid,
            stream_epoch=f"ep-{nid}",
            instance_id=f"inst-{nid}",
            backends=["docker"],
        )
    store.close()

    cfg = _cfg(state_db, runs_root, nodes_yaml=ny)
    client = TestClient(build_admin_app(cfg))
    payload = client.get("/api/overview").json()
    assert payload["nodes_lost"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# Rostered nodes not connected: nodes_lost > 0
# ──────────────────────────────────────────────────────────────────────────────


def test_rostered_not_connected_counts_toward_nodes_lost(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """Two rostered nodes, only one connected → nodes_lost == 1."""
    nodes = ["node-present", "node-lost"]
    ny = _nodes_yaml(tmp_path, nodes)

    store = SqliteStateStore(state_db)
    # Only node-present connects.
    store.record_node_connected(
        node_id="node-present",
        stream_epoch="ep-1",
        instance_id="inst-1",
        backends=["docker"],
    )
    store.close()

    cfg = _cfg(state_db, runs_root, nodes_yaml=ny)
    result = _overview_blocking(cfg, time.time())
    assert result["nodes_lost"] == 1
    assert result["node_connected"] == 1
    assert result["node_rostered"] == 2


def test_all_rostered_disconnected(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """Three rostered nodes, zero connected → nodes_lost == 3."""
    nodes = ["n1", "n2", "n3"]
    ny = _nodes_yaml(tmp_path, nodes)

    # State DB exists but no connected nodes.
    store = SqliteStateStore(state_db)
    store.close()

    cfg = _cfg(state_db, runs_root, nodes_yaml=ny)
    result = _overview_blocking(cfg, time.time())
    assert result["nodes_lost"] == 3
    assert result["node_connected"] == 0
    assert result["node_rostered"] == 3


def test_nodes_lost_via_http_partial_loss(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """Partial loss surfaces correctly via the HTTP endpoint."""
    nodes = ["alive", "gone"]
    ny = _nodes_yaml(tmp_path, nodes)

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        node_id="alive",
        stream_epoch="ep",
        instance_id="i",
        backends=["docker"],
    )
    store.close()

    cfg = _cfg(state_db, runs_root, nodes_yaml=ny)
    client = TestClient(build_admin_app(cfg))
    payload = client.get("/api/overview").json()
    assert payload["nodes_lost"] == 1
    assert payload["node_connected"] == 1
    assert payload["node_rostered"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# Connected non-rostered nodes don't count as lost
# ──────────────────────────────────────────────────────────────────────────────


def test_connected_but_not_rostered_does_not_inflate_nodes_lost(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """A node that connects but is NOT in nodes.yaml is not rostered, so it
    does not contribute to nodes_lost regardless of its connection status."""
    # Roster only node-A; node-B connects but is not rostered.
    ny = _nodes_yaml(tmp_path, ["node-A"])

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        node_id="node-A",
        stream_epoch="ep-A",
        instance_id="inst-A",
        backends=["docker"],
    )
    store.record_node_connected(
        node_id="node-B",
        stream_epoch="ep-B",
        instance_id="inst-B",
        backends=["docker"],
    )
    store.close()

    cfg = _cfg(state_db, runs_root, nodes_yaml=ny)
    result = _overview_blocking(cfg, time.time())
    # node-A is rostered + connected → 0 lost; node-B is connected but
    # not rostered → also doesn't cause nodes_lost.
    assert result["nodes_lost"] == 0
    assert result["node_connected"] == 2
    assert result["node_rostered"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# No nodes.yaml — rostered is empty → nodes_lost is always 0
# ──────────────────────────────────────────────────────────────────────────────


def test_no_nodes_yaml_nodes_lost_is_zero(
    state_db: Path, runs_root: Path,
) -> None:
    """When no nodes.yaml is configured, rostered_ids is empty → nodes_lost=0
    regardless of DB state."""
    store = SqliteStateStore(state_db)
    store.record_node_connected(
        node_id="solo",
        stream_epoch="ep",
        instance_id="i",
        backends=["docker"],
    )
    store.close()

    cfg = _cfg(state_db, runs_root, nodes_yaml=None)
    result = _overview_blocking(cfg, time.time())
    assert result["nodes_lost"] == 0
    assert result["node_rostered"] == 0
    assert result["node_connected"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# Disconnected then reconnected: nodes_lost corrects itself
# ──────────────────────────────────────────────────────────────────────────────


def test_reconnected_node_no_longer_counts_as_lost(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """A node that disconnects then reconnects must show nodes_lost=0 again."""
    ny = _nodes_yaml(tmp_path, ["n1"])

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        node_id="n1",
        stream_epoch="ep1",
        instance_id="i1",
        backends=["docker"],
    )
    store.record_node_disconnected("n1")
    # Reconnect.
    store.record_node_connected(
        node_id="n1",
        stream_epoch="ep2",
        instance_id="i2",
        backends=["docker"],
    )
    store.close()

    cfg = _cfg(state_db, runs_root, nodes_yaml=ny)
    result = _overview_blocking(cfg, time.time())
    assert result["nodes_lost"] == 0
    assert result["node_connected"] == 1
