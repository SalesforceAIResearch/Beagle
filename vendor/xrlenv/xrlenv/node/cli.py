"""``xrlenv-node`` console script.

Boots a :class:`NodeAgent` with the local Docker driver and connects out to
the control plane via :class:`NodeGrpcLink` (spec 21).

Usage::

    xrlenv-node serve --control-plane localhost:50051 --node-id local-A

Run from a separate process / VM than the control plane. The link
auto-reconnects on transient failures with exponential backoff.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xrlenv
from xrlenv.backends.base import SandboxBackend
from xrlenv.backends.docker import DockerBackend, DockerBackendConfig, _default_stub_transport
from xrlenv.node.agent import NodeAgent, NodeAgentConfig
from xrlenv.node.grpc_link import NodeGrpcLink
from xrlenv.node.trajectory_reader import JsonlTrajectoryReader
from xrlenv.observability.logging import configure_logging

if TYPE_CHECKING:
    from xrlenv.node.image_cache import ImageCacheConfig

LOGGER = logging.getLogger("xrlenv.node")


def _image_cache_config_from_env() -> ImageCacheConfig | None:
    """Build an :class:`ImageCacheConfig` from operator environment
    overrides, or ``None`` to use the library defaults.

    Issue #18 — ``XRLENV_PULL_CONCURRENCY``: how many distinct images
    this node-agent pulls concurrently. The library default (2) suits
    image-reuse-heavy RL training; cold-pull-heavy workloads (a
    unique multi-GB image per task) want it higher so the network
    link isn't left idle between pulls. The deploy scripts stamp this
    into ``/etc/xrlenv/node.env``; an operator tunes it there and
    restarts the node-agent. Unset / blank / non-positive / non-int
    → ``None`` (library default), with a warning for the malformed
    cases so a typo doesn't silently no-op.
    """
    import os

    from xrlenv.node.image_cache import ImageCacheConfig

    def _positive_int_env(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return None
        try:
            n = int(raw)
        except ValueError:
            LOGGER.warning(
                "%s=%r is not an integer; using the default", name, raw,
            )
            return None
        if n < 1:
            LOGGER.warning("%s=%d must be >= 1; using the default", name, n)
            return None
        return n

    # AIMD pull-concurrency knobs: floor / ceiling / initial. The
    # limiter adapts between floor and ceiling based on node load.
    overrides: dict[str, Any] = {}
    for env_name, field in (
        ("XRLENV_PULL_CONCURRENCY", "pull_concurrency"),            # floor
        ("XRLENV_PULL_CONCURRENCY_CEILING", "pull_concurrency_ceiling"),
        ("XRLENV_PULL_CONCURRENCY_INITIAL", "pull_concurrency_initial"),
    ):
        val = _positive_int_env(env_name)
        if val is not None:
            overrides[field] = val
            LOGGER.info("image cache: %s=%d (from %s)", field, val, env_name)

    # Eviction headroom caps (GiB). Upper bound on the adaptive reserve
    # (slots x largest_cached_image x safety) so one pathologically large
    # base image can't reserve an unreasonable share of the disk.
    # Specified in GiB for operator friendliness.
    for env_name, field in (
        ("XRLENV_EVICT_THRESHOLD_CAP_GB", "evict_threshold_cap_bytes"),
        ("XRLENV_EVICT_TARGET_CAP_GB", "evict_target_cap_bytes"),
    ):
        gb = _positive_int_env(env_name)
        if gb is not None:
            overrides[field] = gb * 1024**3
            LOGGER.info(
                "image cache: %s=%d GiB (from %s)", field, gb, env_name,
            )

    # I/O-saturation pull throttle. The watermarks are given as percentages
    # (0-100) for operator friendliness; stored as fractions. The node
    # discovers the volume's real IOPS/throughput ceiling by observing
    # util pegged near 100 % — no provisioned-IOPS number is assumed.
    def _pct_env(name: str) -> float | None:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return None
        try:
            pct = float(raw)
        except ValueError:
            LOGGER.warning("%s=%r is not a number; using the default", name, raw)
            return None
        if not 0.0 < pct <= 100.0:
            LOGGER.warning(
                "%s=%g must be in (0, 100]; using the default", name, pct,
            )
            return None
        return pct / 100.0

    for env_name, field in (
        ("XRLENV_IO_UTIL_HIGH_PCT", "io_util_high"),
        ("XRLENV_IO_UTIL_LOW_PCT", "io_util_low"),
    ):
        frac = _pct_env(env_name)
        if frac is not None:
            overrides[field] = frac
            LOGGER.info(
                "image cache: %s=%.2f (from %s)", field, frac, env_name,
            )

    io_raw = os.environ.get("XRLENV_IO_THROTTLE")
    if io_raw is not None and io_raw.strip():
        io_val = io_raw.strip().lower()
        if io_val in ("0", "false", "no", "off"):
            overrides["io_throttle_enabled"] = False
            LOGGER.info("image cache: io_throttle_enabled=False (from XRLENV_IO_THROTTLE)")
        elif io_val in ("1", "true", "yes", "on"):
            overrides["io_throttle_enabled"] = True
        else:
            LOGGER.warning(
                "XRLENV_IO_THROTTLE=%r is not a recognised boolean; "
                "leaving the I/O throttle enabled (the default)", io_raw,
            )

    # io_util_low must not exceed io_util_high (the config validates the
    # full object; pre-check here so a lone low override against the
    # default high gives a clear warning instead of a construction error).
    if (
        "io_util_low" in overrides
        and overrides["io_util_low"] > overrides.get("io_util_high", ImageCacheConfig().io_util_high)
    ):
        LOGGER.warning(
            "XRLENV_IO_UTIL_LOW_PCT (%.2f) exceeds the high watermark; "
            "dropping it (using the default low)", overrides["io_util_low"],
        )
        del overrides["io_util_low"]

    if not overrides:
        return None
    return ImageCacheConfig(**overrides)


def _nonneg_int_env(name: str) -> int | None:
    """Parse a non-negative integer env override (``0`` is meaningful —
    it disables the relevant per-node concurrency cap). Unset / blank /
    malformed / negative → ``None`` (use the library default), warning on
    the malformed cases."""
    import os

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        n = int(raw)
    except ValueError:
        LOGGER.warning("%s=%r is not an integer; using the default", name, raw)
        return None
    if n < 0:
        LOGGER.warning("%s=%d must be >= 0; using the default", name, n)
        return None
    return n


# Per-node raw-container concurrency-cap env overrides → NodeAgentConfig fields.
# Bounding destroys keeps a burst of teardowns from saturating the EBS volume;
# bounding creates limits concurrent image extraction; the sysbox-specific create
# cap is tighter because sysbox-fs pre-register is far slower than a plain runc
# create; bounding archive transfers caps bulk container⇄node tar copies. ``0``
# disables a cap (the sysbox cap then falls back to the general create cap).
# Module-level + parsed by ``_raw_concurrency_overrides`` so the env↔field mapping
# is unit-testable without constructing a full NodeAgent.
_RAW_CONCURRENCY_ENV_FIELDS: tuple[tuple[str, str], ...] = (
    ("XRLENV_RAW_DESTROY_CONCURRENCY", "raw_destroy_concurrency"),
    ("XRLENV_RAW_CREATE_CONCURRENCY", "raw_create_concurrency"),
    ("XRLENV_RAW_SYSBOX_CREATE_CONCURRENCY", "raw_sysbox_create_concurrency"),
    ("XRLENV_RAW_SYSBOX_DESTROY_CONCURRENCY", "raw_sysbox_destroy_concurrency"),
    ("XRLENV_RAW_ARCHIVE_CONCURRENCY", "raw_archive_concurrency"),
    # Plane-split guardrail: max bytes a single get_archive may relay through the
    # control plane before it's refused (ArchiveTooLarge). ``0`` disables.
    ("XRLENV_MAX_GET_ARCHIVE_RELAY_BYTES", "raw_max_get_archive_relay_bytes"),
)


def _raw_concurrency_overrides() -> dict[str, int]:
    """Read the raw-container concurrency-cap env overrides into a
    ``NodeAgentConfig(**kwargs)`` mapping. Only vars that are set (and parse to a
    non-negative int) appear — an unset var leaves the library default in place."""
    overrides: dict[str, int] = {}
    for env_name, field in _RAW_CONCURRENCY_ENV_FIELDS:
        v = _nonneg_int_env(env_name)
        if v is not None:
            overrides[field] = v
            LOGGER.info("node agent: %s=%d (from %s)", field, v, env_name)
    return overrides


def _build_node_agent(
    *,
    node_id: str,
    runs_root: Path,
) -> NodeAgent:
    from xrlenv.control.template_discovery import (
        find_entry_point_manifest_files,
        find_external_template_dir_manifests,
        find_plugin_root,
    )
    from xrlenv.node.image_cache import ImageCacheManager
    from xrlenv.node.image_pins import DEFAULT_PIN_FILE, load_image_pins

    xrlenv_pkg_path = Path(xrlenv.__file__).resolve().parent
    # D22 — discover external plug-in roots from the same env-var +
    # entry-point inputs the control plane consumes for manifest
    # registration. Each root mounts read-only into every sandbox so
    # the in-sandbox stub's ``env_setup`` can import the adapter
    # natively. Single source of truth: an operator who configures
    # XRLENV_TEMPLATE_DIRS or pip-installs an entry-point gets both
    # manifest registration *and* sandbox-side import for free.
    in_tree_root = find_plugin_root(xrlenv_pkg_path)
    external_manifests = (
        find_external_template_dir_manifests()
        + find_entry_point_manifest_files()
    )
    extra_plugin_roots = tuple(sorted({
        m.plugin_root for m in external_manifests
        if m.plugin_root is not None and m.plugin_root != in_tree_root
    }))
    if extra_plugin_roots:
        # D22 startup audit: spec-19 promises the operator-visible
        # signal for platform-injected mounts. The structured per-mount
        # audit row (plugin_root.mounted in the state-store audit
        # table) is phase-1.5 follow-on; today we surface the same
        # information as a structured INFO log line so the node-agent
        # journal carries the inventory.
        for idx, root in enumerate(extra_plugin_roots):
            LOGGER.info(
                "plugin_root.mounted host_path=%s container_target=/opt/xrlenv-extras/%d ro=true",
                root, idx,
            )
    docker = DockerBackend(
        DockerBackendConfig(
            runs_root=runs_root,
            xrlenv_pkg_path=xrlenv_pkg_path,
            xrlenv_plugins_path=in_tree_root,
            extra_plugin_roots=extra_plugin_roots,
            stub_transport=_default_stub_transport(),
        ),
    )
    backends: dict[str, SandboxBackend] = {"docker": docker}
    # Wire an ImageCacheManager so the admin /images view can surface
    # this node's image inventory (operator-reported regression
    # 2026-05-04: pre-fix the node bootstrap omitted this kwarg, so
    # NodeAgent.report_images() returned the empty-fallback default
    # — admin rendered "free disk: 0.00 GiB / Cache is empty" for
    # every connected node even when they had a full image set).
    # Mirrors the wiring xrlenv.control.runtime.build_local_runtime
    # already does for in-process runtimes; the operator pin file
    # uses the same default path (/etc/xrlenv/image-pins.yaml) so a
    # single config covers both deployment shapes.
    pin_set = load_image_pins(DEFAULT_PIN_FILE)
    from xrlenv.node.disk_io import DiskIoSampler
    from xrlenv.node.image_cache import ImageCacheConfig

    cache_cfg = _image_cache_config_from_env()
    eff_cfg = cache_cfg or ImageCacheConfig()
    # I/O-aware pull throttle: watch the docker data-root volume's
    # saturation so cold pulls back off before they peg the EBS volume and
    # wedge containerd's teardown path. Disabled (None) when the operator
    # turns it off; the sampler itself fail-opens if /sys is unavailable.
    disk_io_sampler = (
        DiskIoSampler(
            path_provider=docker.disk_monitor_path,
            high=eff_cfg.io_util_high,
            low=eff_cfg.io_util_low,
            min_interval_s=eff_cfg.io_sample_min_interval_s,
        )
        if eff_cfg.io_throttle_enabled
        else None
    )
    image_cache = ImageCacheManager(
        backend=docker, pins=pin_set,
        config=cache_cfg,
        disk_io_sampler=disk_io_sampler,
    )
    node_cfg_kwargs: dict[str, Any] = {
        "node_id": node_id,
        "backends": backends,
    }
    # Per-node concurrency caps for raw container create/destroy/archive.
    # Operators tune via env; ``0`` disables a cap. Defaults come from
    # NodeAgentConfig. See ``_RAW_CONCURRENCY_ENV_FIELDS``.
    node_cfg_kwargs.update(_raw_concurrency_overrides())
    return NodeAgent(
        NodeAgentConfig(**node_cfg_kwargs),
        image_cache=image_cache,
        trajectory_reader=JsonlTrajectoryReader(runs_root),
    )


async def _serve(args: argparse.Namespace) -> None:
    import os

    runs_root = Path(args.runs_root).expanduser()
    runs_root.mkdir(parents=True, exist_ok=True)
    agent = _build_node_agent(node_id=args.node_id, runs_root=runs_root)
    bearer_token = os.environ.get("XRLENV_NODE_TOKEN") or None

    # Spec 09 GC layer 2: sweep orphan sandboxes before the gRPC link
    # opens, so the control plane never schedules onto a host that still
    # has stale containers competing for the cgroup budget. The previous
    # node-agent process may have been OOM-killed or systemd-restarted
    # mid-rollout; those containers carry the xrlenv.sandbox_id label
    # but no live process owns them.
    try:
        reaped = await agent.gc_orphans()
        if reaped:
            LOGGER.info(
                "startup gc: reaped %d orphan sandbox(es): %s",
                len(reaped), reaped,
            )
    except Exception:
        LOGGER.exception("startup gc: gc_orphans failed; continuing anyway")

    link = NodeGrpcLink(
        agent,
        control_addr=args.control_plane,
        bearer_token=bearer_token,
    )
    LOGGER.info(
        "xrlenv-node id=%s connecting to %s (runs_root=%s)",
        args.node_id,
        args.control_plane,
        runs_root,
    )
    try:
        await link.run_forever()
    except asyncio.CancelledError:
        LOGGER.info("node link cancelled; shutting down")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="xrlenv-node")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Connect to a control plane and serve commands")
    serve.add_argument("--control-plane", required=True, help="host:port of the control plane gRPC server")
    serve.add_argument("--node-id", required=True, help="Stable identifier for this node")
    serve.add_argument(
        "--runs-root",
        default="~/.xrlenv/runs",
        help="Per-rollout host directory (default: ~/.xrlenv/runs)",
    )
    serve.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default INFO)",
    )
    serve.add_argument(
        "--log-format",
        choices=("auto", "json", "pretty"),
        default="auto",
        help=(
            "Log output style: 'pretty' (ANSI-colorized) for an operator "
            "watching a terminal, 'json' (spec-08 structured) for "
            "systemd-journal capture on cloud nodes, 'auto' (default) "
            "picks pretty when stdout is a TTY and json otherwise."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    # 'auto' picks JSON under systemd (no TTY) and pretty when an operator
    # tails the daemon by hand from a terminal. Pass --log-format json to
    # lock spec-08 envelopes regardless of where stdout points.
    configure_logging(level=args.log_level, log_format=args.log_format)
    if args.cmd == "serve":
        try:
            asyncio.run(_serve(args))
        except KeyboardInterrupt:
            LOGGER.info("interrupted; exiting")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
