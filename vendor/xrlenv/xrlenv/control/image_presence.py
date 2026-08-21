"""Per-placement image-presence fan-out helper.

Asks every backend-capable node concurrently whether it has an image
locally; returns the ``dict[node_id, bool]`` the scheduler reads as
the image-affinity input to ``Scheduler.place(image_present=...)``.

The same primitive serves both case-1 (template-driven sandbox
admission via :class:`xrlenv.control.admission.AdmissionQueue`) and
case-2/3 (raw-container acquire via
:class:`xrlenv.control.raw_container_service.RawContainerCoordinator`).
Lifted from ``AdmissionQueue._maybe_query_image_presence`` so the two
acquire paths share one code path — same calibration, same per-node
failure handling, same cost model.

See ``docs/technical/scheduling.md`` "Where ``image_present`` comes
from" for the full picture.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)


async def query_image_presence(
    scheduler: Any,
    image: str | None,
    *,
    backend: str | None = None,
    container_runtime: str | None = None,
) -> dict[str, bool] | None:
    """Pre-fetch image presence per backend-capable node.

    Returns a ``{node_id: True/False}`` dict the scheduler consumes
    as the image-affinity input to ``Scheduler.place(image_present=...)``.

    Returns ``None`` (= "skip the affinity bonus") when:

    - The scheduler has ``image_aware_placement=False`` (operator
      opt-out for uniformly-fast-registry deployments).
    - ``image`` is None (Pattern A unresolved manifests; raw
      acquires that didn't supply an image — the latter shouldn't
      happen in practice but we guard anyway).
    - No backend-capable nodes are attached.

    Per-node ``query_image`` failures (RPC timeout, transport
    hiccup, node off-line mid-fan-out) are logged and treated as
    "absent" so a flaky node doesn't poison the placement decision.

    Cost: one ``query_image`` RPC per backend-capable node per
    invocation, fanned out concurrently via ``asyncio.gather``.
    Wall-clock per call is bounded by the slowest single RPC
    (~50 ms cross-LAN per ``docs/technical/scheduling.md``).
    Phase-2 scale work may add a per-node heartbeat-cached
    snapshot to lift this off the hot path.
    """
    # ``image_aware_placement`` is the operator-tunable opt-out flag
    # (``Scheduler.__init__(image_aware_placement=False)`` for
    # uniformly-fast-registry deployments). Older scheduler doubles
    # used in tests don't carry the attribute; treat absence as
    # "image-aware on" so test fixtures don't have to declare it.
    if not getattr(scheduler, "image_aware_placement", True):
        return None
    if not image:
        return None
    from xrlenv.control.defaults import DEFAULT_BACKEND

    effective_backend = backend or DEFAULT_BACKEND
    candidates = [
        n for n in scheduler.nodes
        if effective_backend in n.supported_backends()
    ]
    # §5.3 — narrow to runtime-eligible nodes BEFORE the presence fan-out.
    # Otherwise a non-sysbox node that happens to hold the image would earn
    # an affinity bonus for a sysbox request and steer placement toward a
    # node that can't run it. Only narrows for a non-default runtime, so the
    # ordinary runc path is unchanged. Defensive getattr — node stand-ins
    # that predate ``supported_runtimes`` advertise only runc, so a sysbox
    # request simply won't match them.
    if container_runtime and container_runtime != "runc":
        candidates = [
            n for n in candidates
            if container_runtime in (
                list(getattr(n, "supported_runtimes", lambda: ["runc"])() or [])
                or ["runc"]
            )
        ]
    if not candidates:
        return None

    async def _query(node: Any) -> tuple[str, bool]:
        try:
            result = await node.query_image(image)
            return node.node_id, bool(result.present)
        except Exception:
            LOGGER.exception(
                "image-affinity: query_image failed on node=%s for %s; "
                "treating as absent", node.node_id, image,
            )
            return node.node_id, False

    results = await asyncio.gather(*(_query(n) for n in candidates))
    return dict(results)
