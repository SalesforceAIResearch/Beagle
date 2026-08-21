"""FFD bin-packer (P1.6.b)."""

from __future__ import annotations

import pytest
from xrlenv.control.image_planner import (
    ImageToPlace,
    InsufficientCapacity,
    NodeBudget,
    plan_placements,
)


def _img(ref: str, size_gib: int, *, replication: int = 1, benchmark: str = "b") -> ImageToPlace:
    return ImageToPlace(
        image_ref=ref, size_bytes=size_gib * 1024**3,
        replication=replication, benchmark=benchmark,
    )


def _node(node_id: str, free_gib: int) -> NodeBudget:
    return NodeBudget(
        node_id=node_id, available_bytes=free_gib * 1024**3,  # type: ignore[arg-type]
    )


def test_empty_plan_returns_empty_placement() -> None:
    result = plan_placements([], [])
    assert result.assignments == ()


def test_single_image_single_node() -> None:
    images = [_img("a:1", 5)]
    nodes = [_node("n1", 100)]
    result = plan_placements(images, nodes)
    assert len(result.assignments) == 1
    assert result.assignments[0].image_ref == "a:1"
    assert result.assignments[0].node_id == "n1"


def test_largest_first_pickup() -> None:
    """Big image goes to the most-free node before small images steal slots."""
    images = [_img("small:1", 5), _img("big:1", 60)]
    nodes = [_node("n1", 50), _node("n2", 100)]
    result = plan_placements(images, nodes)
    big = next(a for a in result.assignments if a.image_ref == "big:1")
    small = next(a for a in result.assignments if a.image_ref == "small:1")
    # Big image gets n2 (100 GiB free); small image fits on n1 (45 GiB
    # remaining after big *would* go there) — but big actually went to n2,
    # so n2 has 40 GiB free + n1 has 50 GiB free; small gets n1.
    assert big.node_id == "n2"
    assert small.node_id == "n1"


def test_replication_spread_across_distinct_nodes() -> None:
    images = [_img("a:1", 10, replication=3)]
    nodes = [_node("n1", 100), _node("n2", 100), _node("n3", 100)]
    result = plan_placements(images, nodes)
    assert {a.node_id for a in result.assignments} == {"n1", "n2", "n3"}


def test_insufficient_capacity_raises() -> None:
    images = [_img("a:1", 200)]
    nodes = [_node("n1", 50), _node("n2", 50)]
    with pytest.raises(InsufficientCapacity) as exc_info:
        plan_placements(images, nodes)
    assert exc_info.value.image_ref == "a:1"
    assert exc_info.value.want_replicas == 1
    assert exc_info.value.fit_count == 0


def test_replication_exceeds_node_count_raises() -> None:
    images = [_img("a:1", 10, replication=3)]
    nodes = [_node("n1", 100), _node("n2", 100)]
    with pytest.raises(InsufficientCapacity) as exc_info:
        plan_placements(images, nodes)
    assert exc_info.value.want_replicas == 3
    assert exc_info.value.fit_count == 2


def test_no_nodes_raises_for_first_image() -> None:
    with pytest.raises(InsufficientCapacity):
        plan_placements([_img("a:1", 1)], [])


def test_spread_tie_break_prefers_emptier_node() -> None:
    """Two nodes with equal free disk after the first image lands → the
    second image should go to the node with fewer assignments."""
    images = [_img("a:1", 10), _img("b:1", 10)]
    nodes = [_node("n1", 100), _node("n2", 100)]
    result = plan_placements(images, nodes)
    # First image picks n1 (sort-stable on equal free + 0 count); after
    # placement, n1 has 90 GiB free + 1 image, n2 has 100 GiB free + 0
    # images. n2 wins on the most-free metric for image 2.
    a = next(a for a in result.assignments if a.image_ref == "a:1")
    b = next(a for a in result.assignments if a.image_ref == "b:1")
    assert {a.node_id, b.node_id} == {"n1", "n2"}


def test_per_node_view_groups_correctly() -> None:
    images = [_img("a:1", 10), _img("b:1", 10), _img("c:1", 10)]
    nodes = [_node("n1", 100), _node("n2", 100)]
    result = plan_placements(images, nodes)
    counts = {nid: len(rows) for nid, rows in result.assignments_by_node.items()}
    assert sum(counts.values()) == 3
    # FFD with R=1 and equal-size images splits 2:1 across nodes.
    assert sorted(counts.values()) == [1, 2]
