"""Admin panel page contract (spec 13).

The admin UI is split by operator intent:

- Monitor: ``/``, ``/health``, ``/rollouts``, ``/sandboxes``, ``/nodes``,
  and ``/images/cache``.
- Plan / catalog: ``/capacity`` and ``/images/catalog``.
- Drilldowns: rollout detail, per-node cache detail, and image placement
  detail are linked from tables and are not top-level navigation entries.

The ``ADMIN_PAGE_OWNED_FACTS`` matrix is executable documentation: each
top-level page owns a disjoint set of canonical fact keys. Before adding
a summary fact to a page, update this matrix and its disjointness test.
"""

from xrlenv.admin.server import (
    AdminBindError,
    AdminServer,
    AdminServerConfig,
    build_admin_app,
)

ADMIN_PAGE_OWNED_FACTS: dict[str, frozenset[str]] = {
    "overview": frozenset({
        "node.active_known_count",
        "sandbox.active_count",
        "rollout.running_count",
        "rollout.finished_recent_count",
        "rollout.failed_recent_count",
        "admin.uptime",
    }),
    "health": frozenset({
        "health.node_signals",
        "health.long_running_sessions",
        "health.failure_rate_high",
        "health.evaluated_at",
        "health.check_count",
    }),
    "rollouts": frozenset({
        "rollout.status",
        "rollout.template",
        "rollout.node_id",
        "rollout.step_count",
        "rollout.final_reward",
        "rollout.duration",
        "rollout.age",
    }),
    "sandboxes": frozenset({
        "sandbox.id",
        "sandbox.node_id",
        "sandbox.template",
        "sandbox.image",
        "sandbox.rollout_id",
        "sandbox.backend",
        "sandbox.status",
        "sandbox.age",
    }),
    "nodes": frozenset({
        "node.id",
        "node.connection_status",
        "node.last_seen",
        "node.rostered",
        "node.cloud",
        "node.expected_address",
    }),
    "images_cache": frozenset({
        "image_cache.reachable_nodes",
        "image_cache.node_pressure",
        "image_cache.node_free_disk",
        "image_cache.node_image_count",
        "image_cache.node_bytes_by_tier",
        "image_cache.node_pinned_count",
    }),
    "images_catalog": frozenset({
        "image_catalog.distinct_images",
        "image_catalog.coverage",
        "image_catalog.replicated_bytes",
        "image_catalog.max_single_node_size",
        "image_catalog.in_use_refs",
        "image_catalog.pinned_nodes",
    }),
    "capacity": frozenset({
        "capacity.estimated_max_concurrent",
        "capacity.cpu_cap",
        "capacity.mem_cap",
        "capacity.disk_cap",
        "capacity.binding_constraint",
        "capacity.computed_at",
    }),
}

__all__ = [
    "ADMIN_PAGE_OWNED_FACTS",
    "AdminBindError",
    "AdminServer",
    "AdminServerConfig",
    "build_admin_app",
]
