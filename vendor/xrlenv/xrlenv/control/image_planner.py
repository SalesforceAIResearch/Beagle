"""FFD bin-packer for build-plan placement (P1.6.b).

Given the catalogue of images-to-build (each with a size hint and
required replication factor) and the cluster's per-node disk budgets,
this module computes "image X lands on node Y" assignments. Pure
function; no I/O.

**Algorithm: First-Fit Decreasing (FFD)**

1. Sort images largest-first (by size hint).
2. For each image, pick the R nodes with the most remaining free
   disk that can fit it. Tie-break by spread — prefer the node with
   the fewest already-assigned images so replicas land on diverse
   hosts.
3. Reduce each picked node's remaining budget by the image size.

~22% worse than optimal in the worst case; ~4-5% in practice. At
phase-1 scale (50 nodes x 500 images) it runs in microseconds.

Phase-2 follow-ons (deferred):

- Demand-weighted replication (hot images on more nodes).
- Reactive rebalance when a node fails.
- Fragmentation-aware scoring (best-fit-decreasing variant).
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import NewType

from pydantic import BaseModel, ConfigDict, Field

NodeId = NewType("NodeId", str)


# ──────────────────────────────────────────────────────────────────────────────
# Size-hint → on-disk conversion
# ──────────────────────────────────────────────────────────────────────────────

# A plan's ``size_hint_bytes`` is *compressed* for ``registry-probe`` /
# ``heuristic`` sources (registries report compressed layer sizes), but the
# bin-packer must reserve the *on-disk* footprint or it over-packs and the
# build hits ENOSPC — with the deferral computing zero overflow because every
# image "fits" against the under-sized number. On-disk is the unpacked snapshot
# (~2.7x compressed, measured on these nodes) plus, on the containerd image
# store, the retained compressed content blob — so ~3-4x total. ``cluster-
# reported`` hints are measured post-materialization (already on-disk) and pass
# through. Both multipliers are env-tunable.
_PACK_ONDISK_MULTIPLIER = float(
    os.environ.get("XRLENV_PACK_ONDISK_MULTIPLIER", "3.0"),
)
_PACK_ONDISK_MULTIPLIER_CLUSTER_REPORTED = float(
    os.environ.get("XRLENV_PACK_ONDISK_MULTIPLIER_CLUSTER_REPORTED", "1.0"),
)


def expected_on_disk_bytes(size_hint_bytes: int, size_hint_source: str) -> int:
    """Expected on-disk footprint of an image, for bin-packing.

    Converts a plan ``size_hint_bytes`` to the disk the image actually occupies
    once materialized. ``registry-probe`` / ``heuristic`` hints are compressed
    registry sizes — packing against them raw over-commits the node and the
    build ENOSPCs (deferral sees zero overflow) — so they're inflated by
    ``XRLENV_PACK_ONDISK_MULTIPLIER`` (default ``3.0``: ~2.7x unpack + the
    containerd content copy). ``cluster-reported`` hints are already measured
    on-disk and pass through (``XRLENV_PACK_ONDISK_MULTIPLIER_CLUSTER_REPORTED``,
    default ``1.0``).
    """
    mult = (
        _PACK_ONDISK_MULTIPLIER_CLUSTER_REPORTED
        if size_hint_source == "cluster-reported"
        else _PACK_ONDISK_MULTIPLIER
    )
    return int(size_hint_bytes * mult)


class ImageToPlace(BaseModel):
    """One image the planner is asked to place.

    ``size_bytes`` is the builder's static ``IMAGE_SIZE_HINT_BYTES`` —
    a conservative upper bound used pre-build. Reality is measured
    after the build and the snapshot updated; the planner doesn't
    re-run automatically.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_ref: str
    size_bytes: int
    replication: int = 1
    benchmark: str = ""
    """Benchmark name carried alongside so the dispatch layer knows
    which builder to load on the node. Empty when the planner is
    invoked without a manifest in scope (rare; mostly tests)."""


class NodeBudget(BaseModel):
    """One node's available disk after operator-configured reservations.

    ``available_bytes`` is the result of:

        total_capacity - reserved_runtime - buffer - already_used_outside_plan

    See :func:`xrlenv.control.build_plan.BuildBudget` for the YAML knobs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: NodeId
    available_bytes: int


class PlanAssignment(BaseModel):
    """One ``(image, node)`` placement decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_ref: str
    node_id: NodeId
    benchmark: str
    size_bytes: int


class PlacementResult(BaseModel):
    """Output of :func:`plan_placements`.

    ``assignments_by_node`` is the canonical view (one list per node);
    ``assignments`` is the flat list for iteration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    assignments: tuple[PlanAssignment, ...]
    assignments_by_node: dict[NodeId, tuple[PlanAssignment, ...]] = Field(
        default_factory=dict,
    )


class InsufficientCapacity(RuntimeError):
    """Raised when no valid placement exists for at least one image.

    Carries the failing image plus how many replica slots it needed
    vs how many nodes had room. The coordinator surfaces this to the
    operator with a message pointing at the budget knobs in
    ``build-plan.yaml``.
    """

    def __init__(
        self, image_ref: str, *, want_replicas: int, fit_count: int,
    ) -> None:
        self.image_ref = image_ref
        self.want_replicas = want_replicas
        self.fit_count = fit_count
        super().__init__(
            f"image {image_ref!r}: replication={want_replicas} but only "
            f"{fit_count} node(s) have room after the budget reservations. "
            f"Increase node disk, drop the budget reservations, or split "
            f"the plan across more nodes.",
        )


class OpportunisticPlacementResult(BaseModel):
    """Output of :func:`plan_opportunistic_placements` (P1.6.g).

    Splits images into ``placed`` (fit within current budget; will be
    pre-built at apply time) and ``deferred`` (no node has room right
    now; recorded as ``registered`` and lazy-built on first
    ``ensure_present``). Deferred entries carry a ``preferred_home``
    node id — the bin-packer's first-choice node — so the image-
    affinity scheduler can route rollouts that need them to that
    node by default (F5=2).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    placed: PlacementResult
    deferred: tuple[DeferredAssignment, ...]


class DeferredAssignment(BaseModel):
    """One ``(image, preferred_home)`` row that didn't fit at apply
    time but stays as the operator's intended placement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_ref: str
    preferred_home: NodeId
    benchmark: str
    size_bytes: int


def plan_placements(
    images: list[ImageToPlace],
    nodes: list[NodeBudget],
) -> PlacementResult:
    """Greedy First-Fit Decreasing placement.

    Returns a :class:`PlacementResult` mapping each image to its R
    chosen nodes. Raises :class:`InsufficientCapacity` if any image
    can't be placed at the requested replication.

    The function is pure: it does not consult or mutate any node-
    side state. Inputs include the *current* budget; concurrent
    plans submitted while one is in_flight aren't supported in
    phase-A (fork E (1) — second apply is rejected).
    """
    if not images:
        return PlacementResult(assignments=())
    if not nodes:
        raise InsufficientCapacity(
            images[0].image_ref, want_replicas=images[0].replication, fit_count=0,
        )

    # Sort images largest-first so big ones get the easy slots.
    images_sorted = sorted(images, key=lambda i: -i.size_bytes)

    # Mutable budget + per-node assignment count for the spread tie-break.
    free: dict[NodeId, int] = {n.node_id: n.available_bytes for n in nodes}
    count_per_node: dict[NodeId, int] = defaultdict(int)
    by_node: dict[NodeId, list[PlanAssignment]] = defaultdict(list)
    flat: list[PlanAssignment] = []

    for img in images_sorted:
        # Candidates: nodes with enough free disk, sorted most-free
        # first; tie-break on the spread heuristic (fewer already-
        # assigned images preferred).
        candidates = sorted(
            (n.node_id for n in nodes if free[n.node_id] >= img.size_bytes),
            key=lambda nid: (-free[nid], count_per_node[nid], nid),
        )
        picks = candidates[: img.replication]
        if len(picks) < img.replication:
            raise InsufficientCapacity(
                img.image_ref,
                want_replicas=img.replication, fit_count=len(picks),
            )
        for nid in picks:
            assignment = PlanAssignment(
                image_ref=img.image_ref, node_id=nid,
                benchmark=img.benchmark, size_bytes=img.size_bytes,
            )
            flat.append(assignment)
            by_node[nid].append(assignment)
            free[nid] -= img.size_bytes
            count_per_node[nid] += 1

    return PlacementResult(
        assignments=tuple(flat),
        assignments_by_node={nid: tuple(rows) for nid, rows in by_node.items()},
    )


def plan_opportunistic_placements(
    images: list[ImageToPlace],
    nodes: list[NodeBudget],
) -> OpportunisticPlacementResult:
    """FFD pass that places what fits + records the rest as deferred.

    Same largest-first / most-free-first / spread-tie-break heuristic
    as :func:`plan_placements`, but never raises
    :class:`InsufficientCapacity`. Images that don't fit anywhere
    become :class:`DeferredAssignment` entries with a
    ``preferred_home`` (the most-free node at the time the image was
    considered) so the image-affinity scheduler can still route
    rollouts that need them to a sensible default node.

    The lazy-build hook (P1.6.g step 1) materializes deferred images
    via the benchmark builder when ``ensure_present`` first fires
    for them at rollout time.
    """
    if not images:
        return OpportunisticPlacementResult(
            placed=PlacementResult(assignments=()), deferred=(),
        )
    if not nodes:
        # No nodes connected — every image is deferred with no
        # preferred home (operator must connect a node before any
        # build can run).
        return OpportunisticPlacementResult(
            placed=PlacementResult(assignments=()),
            deferred=tuple(
                DeferredAssignment(
                    image_ref=i.image_ref, preferred_home="",  # type: ignore[arg-type]
                    benchmark=i.benchmark, size_bytes=i.size_bytes,
                )
                for i in images
            ),
        )

    images_sorted = sorted(images, key=lambda i: -i.size_bytes)
    free: dict[NodeId, int] = {n.node_id: n.available_bytes for n in nodes}
    count_per_node: dict[NodeId, int] = defaultdict(int)
    by_node: dict[NodeId, list[PlanAssignment]] = defaultdict(list)
    placed_flat: list[PlanAssignment] = []
    deferred: list[DeferredAssignment] = []

    for img in images_sorted:
        # Pick R distinct nodes that fit; tie-break by most-free-first
        # then spread (fewer already-assigned images preferred).
        candidates = sorted(
            (n.node_id for n in nodes if free[n.node_id] >= img.size_bytes),
            key=lambda nid: (-free[nid], count_per_node[nid], nid),
        )
        if len(candidates) >= img.replication:
            picks = candidates[: img.replication]
            for nid in picks:
                assignment = PlanAssignment(
                    image_ref=img.image_ref, node_id=nid,
                    benchmark=img.benchmark, size_bytes=img.size_bytes,
                )
                placed_flat.append(assignment)
                by_node[nid].append(assignment)
                free[nid] -= img.size_bytes
                count_per_node[nid] += 1
        else:
            # Doesn't fit at the requested replication. Pick a
            # preferred-home (the most-free node, regardless of fit)
            # so image-affinity scheduling has a default target. The
            # row will be lazy-built on first rollout.
            preferred = max(free, key=lambda n: free[n])
            deferred.append(DeferredAssignment(
                image_ref=img.image_ref,
                preferred_home=preferred,
                benchmark=img.benchmark,
                size_bytes=img.size_bytes,
            ))
            count_per_node[preferred] += 1

    return OpportunisticPlacementResult(
        placed=PlacementResult(
            assignments=tuple(placed_flat),
            assignments_by_node={nid: tuple(rows) for nid, rows in by_node.items()},
        ),
        deferred=tuple(deferred),
    )


__all__ = [
    "DeferredAssignment",
    "ImageToPlace",
    "InsufficientCapacity",
    "NodeBudget",
    "OpportunisticPlacementResult",
    "PlacementResult",
    "PlanAssignment",
    "plan_opportunistic_placements",
    "plan_placements",
]
