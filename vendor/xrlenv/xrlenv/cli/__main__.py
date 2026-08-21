"""``xrlenv`` operator CLI dispatcher (Slice 5b, spec 09).

Phase-0 subcommands:

- ``xrlenv up`` — boot the control plane (DistributedRuntime + /metrics +
  JSON logs) and block until SIGINT.
- ``xrlenv nodes`` — list nodes from ``nodes.yaml`` cross-referenced
  against active sandboxes in ``state.db``.
- ``xrlenv rollouts [--status ...] [--template ...] [--since 5m]`` —
  list rollouts with filters.
- ``xrlenv replay <rollout_id>`` — print the sealed trajectory.
- ``xrlenv events [--since 5m] [--rollout <id>]`` — events log.
- ``xrlenv tail <rollout_id>`` — follow ``trajectory.jsonl`` live.
- ``xrlenv attach <rollout_id>`` — read-only snapshot + tail
  ``coordinator.log``.

Mutating commands (``drain``, ``reload``) need an admin RPC against the
running control plane; they ship in a follow-up slice paired with that
RPC.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from contextlib import suppress
from pathlib import Path

from xrlenv import __version__
from xrlenv.cli.commands import (
    DEFAULT_NODES_YAML,
    DEFAULT_RUNS_ROOT,
    DEFAULT_STATE_DB,
    cmd_attach,
    cmd_audit,
    cmd_db_prune,
    cmd_db_vacuum,
    cmd_events,
    cmd_fairshare_set,
    cmd_fairshare_show,
    cmd_image_evict,
    cmd_images,
    cmd_nodes,
    cmd_replay,
    cmd_rollouts,
    cmd_stub_runtime_layer,
    cmd_tail,
    cmd_tokens_issue,
    cmd_tokens_list,
    cmd_tokens_revoke,
    cmd_tokens_rotate,
    cmd_warmup,
)
from xrlenv.cli.slurm_nodes import cmd_nodes_from_slurm
from xrlenv.node.image_pins import DEFAULT_PIN_FILE
from xrlenv.observability.logging import configure_logging

LOGGER = logging.getLogger("xrlenv.cli")


# ──────────────────────────────────────────────────────────────────────────────
# `up` — boot the control plane
# ──────────────────────────────────────────────────────────────────────────────


async def _serve_control_plane(args: argparse.Namespace) -> int:
    # Local import so CLI subcommands don't pay the gRPC import cost.
    from xrlenv.control.capacity import AimdConfig
    from xrlenv.control.distributed_runtime import build_distributed_runtime

    # 0 disables the optional servers (admin is on by default at 8080;
    # operators who don't want it pass --admin-port 0).
    admin_port: int | None = args.admin_port if args.admin_port else None
    metrics_port: int | None = args.metrics_port if args.metrics_port else None

    runtime = await build_distributed_runtime(
        grpc_host=args.grpc_host,
        grpc_port=args.grpc_port,
        runs_root=Path(args.runs_root).expanduser(),
        state_db_path=Path(args.state_db).expanduser(),
        metrics_host=args.metrics_host,
        metrics_port=metrics_port,
        run_dir_retention_days=args.retention_days,
        audit_retention_days=args.audit_retention_days or None,
        events_retention_days=args.events_retention_days or None,
        raw_rollout_retention_days=args.raw_rollout_retention_days or None,
        admin_host=args.admin_host,
        admin_port=admin_port,
        admin_allow_public=args.admin_allow_public,
        admin_nodes_yaml=(
            Path(args.admin_nodes_yaml).expanduser()
            if args.admin_nodes_yaml else None
        ),
        admin_rollout_page_size=args.admin_rollout_page_size,
        scheduler_max_runs_per_task=args.max_runs_per_task,
        adaptive_admission=args.adaptive_admission,
        # Built unconditionally (cheap); only consulted when
        # --adaptive-admission is set.
        aimd_config=AimdConfig(
            initial_limit=args.aimd_initial_limit,
            p95_bad_threshold_ms=args.aimd_p95_threshold_s * 1000.0,
            max_limit=args.aimd_max_limit,
        ),
    )
    await runtime.start()
    LOGGER.info(
        "xrlenv up: control plane listening on grpc=%s:%d metrics=%s admin=%s",
        args.grpc_host, args.grpc_port,
        f"{args.metrics_host or '127.0.0.1'}:{metrics_port}"
        if metrics_port is not None else "off",
        f"http://{args.admin_host or '127.0.0.1'}:{admin_port}"
        if admin_port is not None else "off",
    )
    stop = asyncio.Event()

    def _handle_signal() -> None:
        LOGGER.info("xrlenv up: signal received; initiating graceful shutdown")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Windows does not support add_signal_handler; rely on KeyboardInterrupt.
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handle_signal)

    try:
        await stop.wait()
    finally:
        await runtime.shutdown()
        LOGGER.info("xrlenv up: shutdown complete")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xrlenv")
    parser.add_argument(
        "-V", "--version", action="version", version=__version__,
    )
    parser.add_argument(
        "--state-db",
        default=str(DEFAULT_STATE_DB),
        help=f"Path to the control-plane state.db (default {DEFAULT_STATE_DB})",
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help=f"Per-rollout artifact root (default {DEFAULT_RUNS_ROOT})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up", help="Boot the control plane")
    # Logging is configured per-invocation, but only the long-running `up`
    # daemon needs to tune it: verbosity/format for a server you watch, and a
    # rotating log file so a process-supervisor stdout capture (Slurm
    # --output, nohup) doesn't grow without bound. One-shot subcommands use the
    # configure_logging() defaults (INFO / auto / no file), so these flags live
    # on `up` rather than globally — keeping `xrlenv up --log-file …` the
    # natural, correct order.
    up.add_argument(
        "--log-level", default="INFO",
        help="Python logging level (default INFO)",
    )
    up.add_argument(
        "--log-format", choices=("auto", "json", "pretty"), default="auto",
        help=(
            "Log output style: 'pretty' for ANSI-colorized terminal output, "
            "'json' for spec-08 structured records, 'auto' (default) picks "
            "pretty when stdout is a TTY and json when piped/redirected."
        ),
    )
    up.add_argument(
        "--log-file", default=None,
        help=(
            "Write the structured (JSON) log firehose to this file with "
            "size-based rotation instead of letting it pile up on stdout. "
            "Use for long-running deployments (e.g. 'xrlenv up' under Slurm) "
            "whose --output capture would otherwise grow without bound. The "
            "path is stable across restarts; rotated files get .1/.2/... "
            "suffixes. With this set, stdout keeps only WARNING+ so the "
            "capture stays small and still shows crashes."
        ),
    )
    up.add_argument(
        "--log-max-bytes", type=int, default=50 * 1024 * 1024,
        help=(
            "Max bytes per rotating log file before rollover "
            "(default 50 MiB; only applies with --log-file)."
        ),
    )
    up.add_argument(
        "--log-backup-count", type=int, default=10,
        help=(
            "Number of rotated log files to retain (default 10; only "
            "applies with --log-file). Disk ceiling = max-bytes * (count+1)."
        ),
    )
    up.add_argument(
        "--stdout-log-level", default=None,
        help=(
            "Minimum level echoed to stdout. Defaults to --log-level when "
            "--log-file is omitted (the full firehose stays on stdout), or "
            "WARNING when --log-file is set (so the stdout/Slurm capture "
            "stays small and shows only crashes). Set e.g. INFO to mirror "
            "the firehose to stdout alongside the rotating file."
        ),
    )
    up.add_argument("--grpc-host", default="127.0.0.1")
    up.add_argument("--grpc-port", type=int, default=50051)
    up.add_argument("--metrics-host", default=None)
    up.add_argument("--metrics-port", type=int, default=9090)
    up.add_argument("--retention-days", type=int, default=14)
    up.add_argument(
        "--audit-retention-days", type=int, default=30,
        help=(
            "Delete audit rows older than N days (spec 20 retention matrix; "
            "0 disables). Audit is the security trail (spec 19) and the dominant "
            "state.db grower."
        ),
    )
    up.add_argument(
        "--events-retention-days", type=int, default=14,
        help="Delete events rows older than N days (0 disables).",
    )
    up.add_argument(
        "--raw-rollout-retention-days", type=int, default=14,
        help="Delete TERMINAL raw_rollouts older than N days (0 disables).",
    )
    up.add_argument(
        "--admin-host", default=None,
        help="Admin panel bind address (default 127.0.0.1)",
    )
    up.add_argument(
        "--admin-port", type=int, default=8080,
        help="Admin panel port; pass 0 to disable",
    )
    up.add_argument(
        "--admin-allow-public", action="store_true",
        help="Allow non-loopback admin bind (spec-19 guard otherwise refuses)",
    )
    up.add_argument(
        "--admin-nodes-yaml", default=None,
        help="Path to nodes.yaml the /nodes view cross-references",
    )
    up.add_argument(
        "--admin-rollout-page-size", type=int, choices=(32, 64, 128, 256), default=32,
        help="Default rows per page in the /rollouts admin view (default 32)",
    )
    up.add_argument(
        "--max-runs-per-task", type=int, default=4,
        help=(
            "Per-node fairness cap on rollouts sharing a `task_key` "
            "(spec 02 anti-affinity). Default 4 matches the phase-0 "
            "GRPO group size. Set lower (e.g. 2) to force the scheduler "
            "to spill same-task_key rollouts onto another node — useful "
            "for acceptance smokes that want deterministic per-node "
            "distribution under `--connect-host` mode."
        ),
    )
    up.add_argument(
        "--adaptive-admission", action="store_true",
        help=(
            "Enable the health-derived adaptive admission controller: "
            "each node's concurrent-acquire limit contracts when its "
            "docker-run latency / error rate degrade and expands when "
            "health holds, so overflow queues instead of melting the "
            "daemon. Off by default (the static estimator). Requires "
            "node-agents on the Stage-1+ build — a node reporting no "
            "health holds at its seed limit."
        ),
    )
    up.add_argument(
        "--aimd-initial-limit", type=int, default=16, metavar="N",
        help=(
            "Adaptive admission slow-start seed: a node's "
            "concurrent-acquire limit before any health data is seen "
            "(default 16). Only used with --adaptive-admission."
        ),
    )
    up.add_argument(
        "--aimd-p95-threshold-s", type=float, default=60.0, metavar="SECONDS",
        help=(
            "Adaptive admission: a node whose docker-run p95 latency "
            "exceeds this has a 'bad' tick and contracts its limit "
            "(default 60s). Only used with --adaptive-admission."
        ),
    )
    up.add_argument(
        "--aimd-max-limit", type=int, default=64, metavar="N",
        help=(
            "Adaptive admission runaway-guardrail: a node's limit "
            "never grows past this (default 64). Not a resource "
            "calculation — the real bound is node health. Only used "
            "with --adaptive-admission."
        ),
    )

    nodes = sub.add_parser("nodes", help="List nodes (rostered + active)")
    nodes.add_argument("--nodes-yaml", default=str(DEFAULT_NODES_YAML))
    nodes.add_argument("--format", choices=("text", "json"), default="text")

    nodes_from_slurm = sub.add_parser(
        "nodes-from-slurm",
        help="Render nodes.yaml from a Slurm script's #SBATCH --nodelist",
    )
    nodes_from_slurm.add_argument(
        "--slurm-script",
        required=True,
        help="Path to the Slurm script containing #SBATCH --nodelist/-w.",
    )
    nodes_from_slurm.add_argument(
        "--output",
        required=True,
        help="Destination nodes.yaml path to write.",
    )
    nodes_from_slurm.add_argument(
        "--id-template",
        default="aws-{hostname}",
        help=(
            "Node id template. Fields: {hostname}, {address}. "
            "Default: aws-{hostname}."
        ),
    )
    nodes_from_slurm.add_argument(
        "--address-template",
        default="{address}",
        help=(
            "Address template. Fields: {hostname}, {address}; {address} "
            "converts AWS node-host hostnames to internal-ip. "
            "Default: {address}."
        ),
    )
    nodes_from_slurm.add_argument(
        "--cloud",
        default="aws",
        help="Value for each node's cloud field. Pass empty string to omit.",
    )
    nodes_from_slurm.add_argument(
        "--backend",
        action="append",
        dest="backends",
        default=None,
        help="Backend to include; repeat for multiple. Default: docker.",
    )
    nodes_from_slurm.add_argument(
        "--auth-token-env",
        default="XRLENV_NODE_TOKEN",
        help="auth_token_env value. Pass empty string to omit.",
    )
    nodes_from_slurm.add_argument(
        "--sysbox-node",
        action="append",
        dest="sysbox_nodes",
        default=None,
        metavar="HOSTNAME_OR_ID",
        help=(
            "Mark a node as a Sysbox-pool member (repeat for multiple); "
            "matches the raw hostname (node-host) or the generated id "
            "(aws-node-host). Emits 'sysbox: true' on that entry. Pool "
            "markers already in the destination file are preserved across "
            "regeneration even without this flag."
        ),
    )
    nodes_from_slurm.add_argument(
        "--allowed-runtime",
        action="append",
        dest="allowed_runtimes",
        default=None,
        metavar="RUNTIME",
        help=(
            "Additively permit an OCI runtime override in "
            "policy.allowed_runtimes (repeat for multiple), e.g. "
            "--allowed-runtime sysbox-runc. Never removes an existing entry."
        ),
    )
    nodes_from_slurm.add_argument(
        "--sysbox-max-concurrent",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stamp 'max_concurrent_by_runtime: {sysbox-runc: N}' on every "
            "Sysbox-pool node — the per-node cap on concurrently running "
            "sysbox containers that keeps a create/exec storm from wedging "
            "sysbox-fs (see notes/design-per-node-runtime-concurrency-cap.md). "
            "A per-node value already in the destination file is preserved and "
            "wins over this default. Omit for unlimited (unchanged)."
        ),
    )
    nodes_from_slurm.add_argument(
        "--allowed-host-path",
        action="append",
        dest="allowed_host_paths",
        default=None,
        metavar="HOST_PATH",
        help=(
            "Additively permit a host bind path in policy.allowed_host_paths "
            "(repeat for multiple), e.g. the EvoClaw golden/data-root that the "
            "oracle mounts read-only. Prefix-matched at gate time, so one "
            "shared read-only data-root entry covers every mount under it. "
            "Never removes an existing entry. Prefer this env-driven knob over "
            "a personal absolute path hand-edited into the roster."
        ),
    )

    rollouts = sub.add_parser("rollouts", help="List rollouts")
    rollouts.add_argument("--status", default=None,
                          help="Filter by status (running/finished/failed/cancelled/truncated)")
    rollouts.add_argument("--template", default=None,
                          help="Filter by template name")
    rollouts.add_argument("--since", default=None,
                          help="Show rollouts created within DURATION (e.g. 5m, 2h)")
    rollouts.add_argument("--format", choices=("text", "json"), default="text")

    replay = sub.add_parser("replay", help="Print a sealed trajectory")
    replay.add_argument("rollout_id")
    replay.add_argument("--format", choices=("text", "json"), default="text")

    events = sub.add_parser("events", help="Events log feed")
    events.add_argument("--since", default=None,
                        help="Only events from the last DURATION (e.g. 5m)")
    events.add_argument("--rollout", default=None, dest="rollout_id",
                        help="Filter by rollout id")
    events.add_argument("--format", choices=("text", "json"), default="text")

    audit = sub.add_parser(
        "audit",
        help="Spec-19 audit log feed (auth.token_used, auth.denied, ...)",
    )
    audit.add_argument("--since", default=None,
                       help="Only entries from the last DURATION (e.g. 5m)")
    audit.add_argument("--kind", default=None,
                       help="Filter by event kind (e.g. auth.token_used)")
    audit.add_argument("--role", default=None,
                       help="Filter by role (node / consumer / operator)")
    audit.add_argument("--format", choices=("text", "json"), default="text")

    tail = sub.add_parser("tail", help="Follow trajectory.jsonl live")
    tail.add_argument("rollout_id")
    tail.add_argument("--stop-after", type=float, default=None,
                      help="(Test-only) stop after N seconds")

    attach = sub.add_parser("attach", help="Read-only inspection + log tail")
    attach.add_argument("rollout_id")
    attach.add_argument("--stop-after", type=float, default=None,
                        help="(Test-only) stop after N seconds")

    images = sub.add_parser(
        "images",
        help="Image management: list cached images, plan FFD distribution, evict",
    )
    # Backwards-compat: ``xrlenv images`` (no subcommand) keeps the
    # original "list cached images on this host" behaviour. The new
    # ``xrlenv images plan`` is the operator FFD distribution
    # primitive (P1.7.B.2).
    images.add_argument(
        "--pin-file", default=str(DEFAULT_PIN_FILE),
        help=f"Operator pin list (default {DEFAULT_PIN_FILE}). "
             f"Used by the legacy bare ``xrlenv images`` command.",
    )
    images.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="Output format for the legacy bare ``xrlenv images`` "
             "command. Ignored by ``images plan``.",
    )
    images_sub = images.add_subparsers(
        dest="images_cmd", required=False,
        title="subcommands",
        description=(
            "``xrlenv images`` (no subcommand) lists cached images "
            "on the local host. ``xrlenv images plan`` runs FFD "
            "bin-packing across the cluster (P1.7.B.2)."
        ),
    )
    images_plan = images_sub.add_parser(
        "plan",
        help="Plan FFD-bin-packed image distribution across the cluster",
        description=(
            "Read a ref list, snapshot per-node free disk via "
            "``report_images``, run the existing FFD planner "
            "(``plan_opportunistic_placements``), persist per-ref "
            "preferred_home rows in the StateStore so subsequent "
            "raw-container acquires honor the plan via "
            "``Scheduler.place(preferred_home_node=...)``. Optional "
            "``--eager-prefetch`` dispatches ``EnsurePresentCommand`` "
            "to each preferred_home node so images arrive before "
            "the first acquire."
        ),
    )
    images_plan.add_argument(
        "--refs", default=None,
        help="Path to a ref list file. One ref per line; lines may "
             "carry an optional ``\\t<size_bytes>`` suffix for the "
             "size hint (defaults to 3 GiB when missing). Lines "
             "starting with '#' are comments.",
    )
    images_plan.add_argument(
        "--ref", action="append", default=[],
        help="Ref to plan (repeatable). Mutually exclusive with --refs.",
    )
    images_plan.add_argument(
        "--default-size", type=int, default=3 * 1024 * 1024 * 1024,
        help="Default size hint in bytes when a ref has no inline "
             "size (3 GiB by default — sized for the SWE-bench "
             "Verified per-instance images).",
    )
    images_plan.add_argument(
        "--eager-prefetch", action="store_true",
        help="After planning, dispatch EnsurePresentCommand to each "
             "preferred_home node so images arrive before the first "
             "acquire. Default: lazy — node pulls on first acquire.",
    )
    images_plan.add_argument(
        "--control-host", default="127.0.0.1",
        help="Control-plane gRPC host (default 127.0.0.1).",
    )
    images_plan.add_argument(
        "--control-port", type=int, default=50051,
        help="Control-plane gRPC port (default 50051).",
    )
    images_plan.add_argument(
        "--operator-token", default=None,
        help="Operator bearer token (or set $XRLENV_OPERATOR_TOKEN).",
    )

    images_evict = images_sub.add_parser(
        "evict",
        help="Evict an image from every node's cache so the next acquire re-pulls",
        description=(
            "Fan an eviction out to every connected node: each node "
            "removes the local image matching <image_ref> (matched "
            "registry-agnostically, so a bare ref matches the "
            "registry-qualified tag a node pulled) so the next acquire "
            "re-pulls fresh from the registry. The escape hatch for the "
            "mutable-tag staleness problem — after a rebuild + re-push "
            "under the same tag a node never re-pulls on its own. In-use "
            "/ pinned images are skipped unless --force."
        ),
    )
    images_evict.add_argument(
        "image_ref",
        help="Image ref to evict (bare or registry-qualified; tag or @digest).",
    )
    images_evict.add_argument(
        "--force", action="store_true",
        help="Evict even in-use / pinned images (docker rmi -f). Default: "
             "skip them so a live rollout is never disrupted.",
    )
    images_evict.add_argument(
        "--connect-host", default=None, dest="connect_host",
        help="Admin server host. Required — evict is a cluster-driven flow.",
    )
    images_evict.add_argument(
        "--connect-port", type=int, default=8080, dest="connect_port",
        help="Admin server port (default 8080).",
    )
    images_evict.add_argument(
        "--operator-token", default=None, dest="operator_token",
        help="Operator-role bearer token for the admin API. Defaults to "
             "$XRLENV_OPERATOR_TOKEN or $XRLENV_HOME/secrets/operator.token.",
    )

    warmup = sub.add_parser(
        "warmup", help="Pre-pull images so they're warm before the next rollout",
    )
    warmup.add_argument("images", nargs="+", help="One or more image refs to pull")
    warmup.add_argument(
        "--deadline", type=float, default=600.0, dest="deadline_s",
        help="Per-image pull timeout in seconds (default 600)",
    )

    stub_runtime = sub.add_parser(
        "stub-runtime",
        help="Build the canonical stub-runtime image layer (D12 stage 1).",
    )
    stub_runtime_sub = stub_runtime.add_subparsers(
        dest="stub_runtime_cmd", required=True,
    )
    stub_runtime_layer = stub_runtime_sub.add_parser(
        "layer",
        help=(
            "Layer the platform's stub-runtime deps "
            "(pydantic+aiohttp+pyyaml, with apt-install python+pip if "
            "missing) on top of an upstream image."
        ),
    )
    stub_runtime_layer.add_argument(
        "--base", required=True,
        help="Upstream image tag to layer on top of "
             "(e.g. terminal-bench-2-base/fix-git:0.1).",
    )
    stub_runtime_layer.add_argument(
        "--out", required=True, dest="out_tag",
        help="Tag for the resulting image "
             "(e.g. terminal-bench-2/fix-git:0.1).",
    )

    build = sub.add_parser(
        "build",
        help=(
            "Apply a build plan (P1.6) — distribute image builds across "
            "the cluster from one operator command. See docs/operator/"
            "build-plans.md for the YAML schema."
        ),
    )
    build_sub = build.add_subparsers(dest="build_cmd", required=True)
    build_apply = build_sub.add_parser(
        "apply",
        help=(
            "Apply a plan. Pass --plan PATH for a YAML; or --benchmark NAME "
            "+ --smoke|--instances|--all to lower an imperative shorthand."
        ),
    )
    build_apply.add_argument(
        "--plan", default=None,
        help="Path to a build-plan.yaml. Mutually exclusive with --benchmark.",
    )
    build_apply.add_argument(
        "--benchmark", default=None,
        help="Imperative shorthand: build for one benchmark (lowered to a "
             "transient plan in the CLI).",
    )
    build_apply.add_argument(
        "--smoke", action="store_true",
        help="With --benchmark: build the benchmark's smoke set.",
    )
    build_apply.add_argument(
        "--instances", default=None,
        help="With --benchmark: comma-separated explicit instance ids.",
    )
    build_apply.add_argument(
        "--all", action="store_true",
        help="With --benchmark: build every instance in the benchmark catalog.",
    )
    build_apply.add_argument(
        "--build-path", default=None,
        help="Plug-in-specific build mode (e.g. 'pull-and-retag' or "
             "'build-locally' for swebench-verified). Ignored unless the "
             "plug-in supports it.",
    )
    build_apply.add_argument(
        "--replication", type=int, default=None,
        help="Override the per-image replication factor (default 1).",
    )
    build_apply.add_argument(
        "--reserved-runtime-gb", type=int, default=30,
        help="Disk reserved per node for running containers (default 30).",
    )
    build_apply.add_argument(
        "--buffer-gb", type=int, default=10,
        help="Margin per node before LRU eviction (default 10).",
    )
    build_apply.add_argument(
        "--build-tarball-max-bytes", type=int, default=None,
        dest="build_tarball_max_bytes",
        help="Cap on tarball-source build-context size (default "
             "100 MB). Operator-side: enforced by the CLI before "
             "any wire traffic — oversized contexts reject with a "
             "clear ``ManifestInvalid`` so you can ``.dockerignore``-"
             "trim and retry without burning cluster cycles. Raise "
             "this when shipping unusually large contexts; never "
             "raise it above the gRPC channel cap (128 MB minus "
             "envelope headroom — the ``BuildImageCommand`` would "
             "fail to deserialize on the node).",
    )
    build_apply.add_argument(
        "--skip-if-present", action="store_true",
        dest="skip_if_present",
        help="For per-image-ref plans (git / tarball entries): "
             "short-circuit the source-build dispatch when the "
             "image is already tagged on the chosen node. The "
             "node returns ``ok`` without cloning, untarring, or "
             "invoking ``docker build``. Use this for warm-cluster "
             "re-applies — typical after ``xrlenv build "
             "calibrate`` (plan_id changes but builds don't) or "
             "to retry a partial-failure plan without rebuilding "
             "the entries that already succeeded. Registry-source "
             "entries already short-circuit via the cache's "
             "``ensure_present`` regardless of this flag. ``--force``"
             " overrides this flag: forced builds always dispatch.",
    )
    build_apply.add_argument(
        "--dry-run", action="store_true",
        help="Print the placement without dispatching builds.",
    )
    build_apply.add_argument(
        "--force", action="store_true",
        help="Rebuild even if the local tag already exists "
             "(propagates to each builder's force=True path).",
    )
    build_apply.add_argument(
        "--eager", action="store_true",
        help=(
            "Pre-build EVERY image at apply time. Fails the apply if "
            "the budget can't fit them all. Default is opportunistic: "
            "pre-build what fits, leave the rest as ``registered`` for "
            "lazy-build via ensure_present at first rollout."
        ),
    )
    build_apply.add_argument(
        "--fill-missing", action="store_true", dest="fill_missing",
        help=(
            "Only build entries that aren't currently present on any "
            "connected node. Queries cluster inventory via "
            "report_images, re-anchors assignment rows for entries "
            "already cached somewhere (no work), and dispatches "
            "ensure_present only for the truly-missing subset. "
            "Use after a partial_failure to retry just the failed "
            "pulls without re-touching the successful rest, or after "
            "eviction has gone through to bring evicted images back. "
            "Cluster-only (--connect-host required); mutually exclusive "
            "with --force and --eager."
        ),
    )
    build_apply.add_argument(
        "--connect-host", default=None, dest="connect_host",
        help="Target a running control plane's admin server instead "
             "of an in-process LocalRuntime. CLI POSTs the plan to "
             "<connect-host>:<connect-port>/api/build/apply, polls "
             "for completion, prints per-image results.",
    )
    build_apply.add_argument(
        "--connect-port", type=int, default=8080, dest="connect_port",
        help="Admin server port (default 8080; matches xrlenv up).",
    )
    build_apply.add_argument(
        "--operator-token", default=None, dest="operator_token",
        help="Operator-role bearer token for the admin API. Defaults "
             "to $XRLENV_OPERATOR_TOKEN or "
             "$XRLENV_HOME/secrets/operator.token "
             "(default ~/.xrlenv/secrets/operator.token).",
    )
    build_apply.add_argument(
        "--concurrency", type=int, default=None, dest="concurrency",
        help="Per-invocation coordinator fan-out: max in-flight image "
             "dispatches across the cluster for THIS apply. Overrides the "
             "XRLENV_BUILD_CONCURRENCY default with no control-plane "
             "restart. Set it to roughly nodes x the per-node pull ceiling "
             "to saturate idle nodes (e.g. 3 nodes x 64 ~= 192).",
    )

    build_status = build_sub.add_parser(
        "status",
        help=(
            "Show the most recent build plan's status. With --plan PLAN_ID "
            "shows that plan's per-assignment rollup. PLAN_ID accepts a "
            "full SHA-256 plan_id or a unique prefix (>=4 chars; matches "
            "the 12-char short id from the admin /builds panel). Without "
            "args, shows the latest applied plan."
        ),
    )
    build_status.add_argument(
        "--plan", default=None, dest="status_plan_id",
        help="Specific plan_id (or unique prefix) to inspect. "
             "Default = latest applied.",
    )

    build_cancel = build_sub.add_parser(
        "cancel",
        help=(
            "Cancel a build plan. With --connect-host, dispatches a "
            "CancelBuildImageCommand to each node currently building "
            "an assignment so any in-flight ``docker build`` is "
            "interrupted (kills the running build container, cancels "
            "the asyncio task) — the recommended path for a live "
            "cluster apply. Without --connect-host, just marks the "
            "plan ``cancelled`` in state.db so the admin panel and "
            "operator polling converge; in-flight builds are NOT "
            "interrupted in this mode (use it for clearing a stuck "
            "plan record after a disconnected apply)."
        ),
    )
    build_cancel.add_argument(
        "--plan", required=True, dest="cancel_plan_id",
        help="plan_id (full SHA-256 or unique prefix >=4 chars) to cancel.",
    )
    build_cancel.add_argument(
        "--connect-host", default=None, dest="connect_host",
        help="Reach the admin server at this host so the cancel "
             "actually interrupts running cluster builds. Without "
             "this flag the cancel is local-only (state.db update; "
             "nodes keep building).",
    )
    build_cancel.add_argument(
        "--connect-port", type=int, default=8080, dest="connect_port",
        help="Admin server port (default 8080).",
    )
    build_cancel.add_argument(
        "--operator-token", default=None, dest="operator_token",
        help="Operator-role bearer token for the admin API. Defaults "
             "to $XRLENV_OPERATOR_TOKEN or "
             "$XRLENV_HOME/secrets/operator.token "
             "(default ~/.xrlenv/secrets/operator.token).",
    )

    build_calibrate = build_sub.add_parser(
        "calibrate",
        help=(
            "Probe each connected node's image cache, take the max "
            "size_bytes per image_ref, and write a calibrated YAML "
            "with size_hint_source=cluster-reported. Run after a "
            "first cluster build to replace heuristic / registry-probe "
            "size hints with measured values; the result feeds back "
            "into the FFD bin-packer for tighter placement on the "
            "next apply."
        ),
    )
    build_calibrate.add_argument(
        "--plan", required=True, dest="calibrate_plan",
        help="Path to the input build-plan.yaml.",
    )
    build_calibrate.add_argument(
        "--output", required=True, dest="calibrate_output",
        help="Where to write the calibrated YAML. Operators commit "
             "this to the canonical plan path once they're happy with "
             "the measurements (review the diff first).",
    )
    build_calibrate.add_argument(
        "--connect-host", default=None, dest="connect_host",
        help="Admin server host. Required — calibrate is an "
             "explicitly cluster-driven flow.",
    )
    build_calibrate.add_argument(
        "--connect-port", type=int, default=8080, dest="connect_port",
        help="Admin server port (default 8080).",
    )
    build_calibrate.add_argument(
        "--operator-token", default=None, dest="operator_token",
        help="Operator-role bearer token for the admin API. Defaults "
             "to $XRLENV_OPERATOR_TOKEN or "
             "$XRLENV_HOME/secrets/operator.token "
             "(default ~/.xrlenv/secrets/operator.token).",
    )

    tokens = sub.add_parser(
        "tokens", help="Issue / inspect bearer tokens (spec 19)",
    )
    tokens_sub = tokens.add_subparsers(dest="tokens_cmd", required=True)
    tokens_issue = tokens_sub.add_parser(
        "issue", help="Issue a fresh bearer token for a role",
    )
    tokens_issue.add_argument(
        "role", choices=("node", "consumer", "operator", "viewer"),
        help="Identity the token authorizes (spec 19 §\"Identities\")",
    )
    tokens_issue.add_argument(
        "--owner", default=None,
        help="Mint a per-user token for this tenant id (e.g. --owner alice). "
             "The control plane stamps every rollout / raw session this token "
             "starts with this owner_id (read off the verified token, not "
             "client-supplied), the admin panel scopes a per-user viewer "
             "token to only this owner's jobs, and follow-up RPCs reject "
             "acting on another owner's work. You can also revoke this user "
             "individually without rotating the shared token. Multiple "
             "per-user tokens of the same role coexist. Omit for the legacy "
             "single shared role-token (owner_id=\"default\"). Not valid with "
             "role=node.",
    )
    tokens_issue.add_argument(
        "--name", default=None,
        help="Optional human label for the owner, shown in `xrlenv tokens "
             "list`. Operator convenience only — does not affect privileges. "
             "Only meaningful with --owner.",
    )
    tokens_issue.add_argument(
        "--secrets-root", default=None,
        help="Override the secrets dir (default $XRLENV_HOME/secrets, "
             "i.e. ~/.xrlenv/secrets)",
    )

    tokens_rotate = tokens_sub.add_parser(
        "rotate",
        help="Issue a fresh token for an existing role; immediate cutover "
             "by default. Pass --grace to keep the previous token live for a "
             "rollover window.",
    )
    tokens_rotate.add_argument(
        "role", choices=("node", "consumer", "operator", "viewer"),
        help="Role whose token will be replaced (spec 19 §\"Identities\")",
    )
    tokens_rotate.add_argument(
        "--grace", default="0",
        help="Keep the previous token valid for this duration after "
             "rotation. Accepts plain seconds (e.g. 3600) or a unit suffix: "
             "s, m, h, d (e.g. 24h). Default 0 — immediate cutover.",
    )
    tokens_rotate.add_argument(
        "--secrets-root", default=None,
        help="Override the secrets dir (default $XRLENV_HOME/secrets, "
             "i.e. ~/.xrlenv/secrets)",
    )

    tokens_revoke = tokens_sub.add_parser(
        "revoke",
        help="Mark a token revoked by its 12-char token_id (or any ≥6-char "
             "unique prefix). Persists to revoked.json so a running control "
             "plane picks the change up on its next hot-reload.",
    )
    tokens_revoke.add_argument(
        "token_id",
        help="The token_id (or unique prefix ≥ 6 chars) of the token to "
             "revoke. Operators normally copy this from a `tokens list` "
             "row or an audit-log digest_hint.",
    )
    tokens_revoke.add_argument(
        "--secrets-root", default=None,
        help="Override the secrets dir (default $XRLENV_HOME/secrets, "
             "i.e. ~/.xrlenv/secrets)",
    )

    tokens_list = tokens_sub.add_parser(
        "list",
        help="Show active token + grace + revocation state for each role.",
    )
    tokens_list.add_argument(
        "--secrets-root", default=None,
        help="Override the secrets dir (default $XRLENV_HOME/secrets, "
             "i.e. ~/.xrlenv/secrets)",
    )

    # ── fairshare — live multi-user fair-share policy ────────────────────────
    fairshare = sub.add_parser(
        "fairshare",
        help="Inspect / tune the live multi-user fair-share policy.",
    )
    fairshare_sub = fairshare.add_subparsers(
        dest="fairshare_cmd", required=True,
    )
    fairshare_show = fairshare_sub.add_parser(
        "show", help="Print the current fair-share policy + per-owner usage.",
    )
    fairshare_show.add_argument(
        "--state-db", default=None,
        help="Path to state.db (default $XRLENV_HOME/state.db, "
             "i.e. ~/.xrlenv/state.db).",
    )
    fairshare_set = fairshare_sub.add_parser(
        "set",
        help=(
            "Tune fair-share live (no restart, running jobs untouched). "
            "Lowering a cap / pausing only stops NEW admissions."
        ),
    )
    fairshare_set.add_argument(
        "--default-cap", type=int, default=None,
        help="Default concurrent-sandbox cap for each owner. Enables fairness. "
             "Omit to leave unchanged.",
    )
    fairshare_set.add_argument(
        "--disable", action="store_true",
        help="Turn fairness OFF (owners run uncapped again).",
    )
    fairshare_set.add_argument(
        "--owner", default=None,
        help="Tenant id to tune (use with --cap/--uncap/--recap/--block/"
             "--unblock).",
    )
    fairshare_set.add_argument(
        "--cap", type=int, default=None,
        help="Set an owner-specific concurrent-sandbox cap.",
    )
    fairshare_set.add_argument(
        "--uncap", action="store_true",
        help="Bypass fair-share caps for --owner; scheduler resources still apply.",
    )
    fairshare_set.add_argument(
        "--recap", action="store_true",
        help="Return --owner to the default cap and clear uncapped/blocked state.",
    )
    fairshare_set.add_argument(
        "--block", action="store_true",
        help="Stop NEW admissions for --owner (running jobs keep going).",
    )
    fairshare_set.add_argument(
        "--unblock", action="store_true",
        help="Resume admissions for --owner.",
    )
    fairshare_set.add_argument(
        "--clear-owner", default=None,
        help=(
            "Remove a tenant's override entirely "
            "(default cap, not uncapped/blocked)."
        ),
    )
    fairshare_set.add_argument(
        "--state-db", default=None,
        help="Path to state.db (default $XRLENV_HOME/state.db, "
             "i.e. ~/.xrlenv/state.db).",
    )

    db = sub.add_parser(
        "db",
        help="state.db maintenance — retention prune + VACUUM (spec 20).",
    )
    db_sub = db.add_subparsers(dest="db_cmd", required=True)
    db_prune = db_sub.add_parser(
        "prune",
        help=(
            "Hard-delete rows past their retention window (spec 20 matrix). The "
            "control plane also does this every 24 h; run this to reclaim on "
            "demand before `db vacuum`."
        ),
    )
    db_prune.add_argument("--state-db", default=str(DEFAULT_STATE_DB))
    db_prune.add_argument("--audit-retention-days", type=int, default=30)
    db_prune.add_argument("--events-retention-days", type=int, default=14)
    db_prune.add_argument("--raw-rollout-retention-days", type=int, default=14)
    db_vacuum = db_sub.add_parser(
        "vacuum",
        help=(
            "VACUUM state.db to return freed pages to the filesystem. Run with "
            "the control plane STOPPED — VACUUM needs exclusive access."
        ),
    )
    db_vacuum.add_argument("--state-db", default=str(DEFAULT_STATE_DB))

    bootstrap = sub.add_parser(
        "bootstrap",
        help=(
            "Install + configure the xrlenv-node daemon on a freshly "
            "provisioned VM. Replaces deploy/bootstrap-{gcp,aws}.sh."
        ),
    )
    bootstrap.add_argument(
        "--target", choices=("gcp", "aws", "linux-generic"),
        required=True,
        help=(
            "Cloud target. Drives docker-install strategy + cloud "
            "metadata auto-detect for --node-id."
        ),
    )
    bootstrap.add_argument(
        "--control-plane", default=None,
        help=(
            "host:port of the control-plane gRPC endpoint. Defaults "
            "to $XRLENV_CONTROL_PLANE."
        ),
    )
    bootstrap.add_argument(
        "--node-id", default=None,
        help=(
            "Stable identifier for this node. Defaults to "
            "$XRLENV_NODE_ID; falls through to GCP metadata "
            "(--target gcp) or AWS IMDSv2 (--target aws) auto-detect."
        ),
    )
    bootstrap.add_argument(
        "--target-os", default=None,
        help=(
            "Operator override for the OS probe. Pass when "
            "/etc/os-release is missing or wrong (custom AMIs). "
            "Accepts the same IDs the probe emits "
            "(amzn / rhel / fedora / ubuntu / debian)."
        ),
    )
    bootstrap.add_argument(
        "--xrlenv-wheel", default=None,
        help="Local wheel path to install instead of the PyPI fallback.",
    )
    bootstrap.add_argument(
        "--xrlenv-repo", default=None,
        help=(
            "Checkout dir containing pyproject.toml; installed "
            "non-editable into /opt/xrlenv/.venv."
        ),
    )
    bootstrap.add_argument(
        "--xrlenv-version", default=None,
        help=(
            "PyPI version pin used when neither --xrlenv-wheel nor "
            "--xrlenv-repo is set. Defaults to $XRLENV_VERSION or "
            "'main'."
        ),
    )
    bootstrap.add_argument(
        "--runtime-user", default=None,
        help=(
            "System user the daemon runs as (default 'xrlenv'). "
            "Operators rarely change this."
        ),
    )
    bootstrap.add_argument(
        "--install-root", default=None,
        help="Install prefix (default /opt/xrlenv).",
    )
    bootstrap.add_argument(
        "--skip-operator-docker-group", action="store_true", default=False,
        help=(
            "Skip adding $SUDO_USER to the docker group. Use for "
            "hardened-security setups or multi-operator hosts."
        ),
    )
    bootstrap.add_argument(
        "--dry-run", action="store_true", default=False,
        help=(
            "Print the planned step sequence + the commands each "
            "step would run. Run this first on a new VM to preview."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    # The logging flags live on the `up` subparser (only the daemon tunes
    # them), so read them defensively: every other subcommand falls back to the
    # configure_logging() defaults — INFO / auto / no rotating file.
    configure_logging(
        level=getattr(args, "log_level", "INFO"),
        log_format=getattr(args, "log_format", "auto"),
        log_file=getattr(args, "log_file", None),
        log_max_bytes=getattr(args, "log_max_bytes", 50 * 1024 * 1024),
        log_backup_count=getattr(args, "log_backup_count", 10),
        stdout_level=getattr(args, "stdout_log_level", None),
    )

    # The `fairshare` subcommands redefine `--state-db` (so it can be given
    # after the subcommand) with a None default, which argparse copies over the
    # top-level default. This resolve runs for *every* subcommand before
    # dispatch, so fall back to DEFAULT_STATE_DB instead of crashing on None.
    state_db = (
        Path(args.state_db).expanduser() if args.state_db else DEFAULT_STATE_DB
    )
    runs_root = Path(args.runs_root).expanduser()

    if args.cmd == "up":
        try:
            return asyncio.run(_serve_control_plane(args))
        except KeyboardInterrupt:
            LOGGER.info("xrlenv up: interrupted")
            return 0
    if args.cmd == "nodes":
        return cmd_nodes(
            state_db=state_db,
            nodes_yaml=Path(args.nodes_yaml).expanduser(),
            output_format=args.format,
            out=sys.stdout,
        )
    if args.cmd == "nodes-from-slurm":
        return cmd_nodes_from_slurm(
            slurm_script=Path(args.slurm_script).expanduser(),
            output=Path(args.output).expanduser(),
            id_template=args.id_template,
            address_template=args.address_template,
            cloud=args.cloud or None,
            backends=args.backends or ["docker"],
            auth_token_env=args.auth_token_env or None,
            sysbox_nodes=args.sysbox_nodes or None,
            allowed_runtimes=args.allowed_runtimes or None,
            sysbox_max_concurrent=args.sysbox_max_concurrent,
            allowed_host_paths=args.allowed_host_paths or None,
            out=sys.stdout,
        )
    if args.cmd == "rollouts":
        return cmd_rollouts(
            state_db=state_db,
            status=args.status,
            template=args.template,
            since=args.since,
            output_format=args.format,
            out=sys.stdout,
        )
    if args.cmd == "replay":
        return cmd_replay(
            args.rollout_id, runs_root=runs_root,
            output_format=args.format, out=sys.stdout,
        )
    if args.cmd == "events":
        return cmd_events(
            state_db=state_db,
            since=args.since,
            rollout_id=args.rollout_id,
            output_format=args.format,
            out=sys.stdout,
        )
    if args.cmd == "audit":
        return cmd_audit(
            state_db=state_db,
            since=args.since,
            kind=args.kind,
            role=args.role,
            output_format=args.format,
            out=sys.stdout,
        )
    if args.cmd == "db":
        if args.db_cmd == "prune":
            return cmd_db_prune(
                state_db=Path(args.state_db).expanduser(),
                audit_retention_days=args.audit_retention_days or None,
                events_retention_days=args.events_retention_days or None,
                raw_rollout_retention_days=args.raw_rollout_retention_days or None,
                out=sys.stdout,
            )
        if args.db_cmd == "vacuum":
            return cmd_db_vacuum(
                state_db=Path(args.state_db).expanduser(),
                out=sys.stdout,
            )
    if args.cmd == "tail":
        return cmd_tail(
            args.rollout_id, runs_root=runs_root,
            stop_after_s=args.stop_after, out=sys.stdout,
        )
    if args.cmd == "attach":
        return cmd_attach(
            args.rollout_id, state_db=state_db, runs_root=runs_root,
            stop_after_s=args.stop_after, out=sys.stdout,
        )
    if args.cmd == "images":
        if getattr(args, "images_cmd", None) == "plan":
            from xrlenv.cli.images_plan_cmd import cmd_images_plan
            return cmd_images_plan(
                refs_file=(
                    Path(args.refs).expanduser() if args.refs else None
                ),
                refs_inline=list(args.ref),
                default_size_bytes=args.default_size,
                eager_prefetch=args.eager_prefetch,
                control_host=args.control_host,
                control_port=args.control_port,
                operator_token=args.operator_token,
                out=sys.stdout,
            )
        if getattr(args, "images_cmd", None) == "evict":
            return cmd_image_evict(
                image_ref=args.image_ref,
                force=args.force,
                connect_host=args.connect_host,
                connect_port=args.connect_port,
                operator_token=args.operator_token,
                out=sys.stdout,
            )
        # No subcommand → legacy "list cached images" behaviour.
        return cmd_images(
            pin_file=Path(args.pin_file).expanduser(),
            output_format=args.format, out=sys.stdout,
        )
    if args.cmd == "warmup":
        return cmd_warmup(
            args.images, deadline_s=args.deadline_s, out=sys.stdout,
        )
    if args.cmd == "stub-runtime" and args.stub_runtime_cmd == "layer":
        return cmd_stub_runtime_layer(
            base=args.base, out_tag=args.out_tag, out=sys.stdout,
        )
    if args.cmd == "build" and args.build_cmd == "status":
        from xrlenv.cli.commands import cmd_build_status

        return cmd_build_status(
            plan_id=args.status_plan_id,
            state_db=state_db, out=sys.stdout,
        )
    if args.cmd == "build" and args.build_cmd == "cancel":
        from xrlenv.cli.commands import cmd_build_cancel

        return cmd_build_cancel(
            plan_id=args.cancel_plan_id,
            state_db=state_db, out=sys.stdout,
            connect_host=args.connect_host,
            connect_port=args.connect_port,
            operator_token=args.operator_token,
        )
    if args.cmd == "build" and args.build_cmd == "calibrate":
        from xrlenv.cli.commands import cmd_build_calibrate

        return cmd_build_calibrate(
            plan_path=Path(args.calibrate_plan).expanduser(),
            output_path=Path(args.calibrate_output).expanduser(),
            out=sys.stdout,
            connect_host=args.connect_host,
            connect_port=args.connect_port,
            operator_token=args.operator_token,
        )
    if args.cmd == "build" and args.build_cmd == "apply":
        from xrlenv.cli.commands import cmd_build_apply

        return cmd_build_apply(
            plan_path=Path(args.plan).expanduser() if args.plan else None,
            benchmark=args.benchmark,
            smoke=args.smoke,
            instances=args.instances,
            all_=args.all,
            build_path=args.build_path,
            replication=args.replication,
            reserved_runtime_gb=args.reserved_runtime_gb,
            buffer_gb=args.buffer_gb,
            tarball_max_bytes=args.build_tarball_max_bytes,
            dry_run=args.dry_run,
            force=args.force,
            eager=args.eager,
            fill_missing=args.fill_missing,
            skip_if_present=args.skip_if_present,
            concurrency=args.concurrency,
            state_db=state_db, runs_root=runs_root,
            connect_host=args.connect_host,
            connect_port=args.connect_port,
            operator_token=args.operator_token,
            out=sys.stdout,
        )
    if args.cmd == "tokens" and args.tokens_cmd == "issue":
        return cmd_tokens_issue(
            args.role,
            owner=args.owner,
            display_name=args.name,
            secrets_root=(
                Path(args.secrets_root).expanduser()
                if args.secrets_root else None
            ),
            out=sys.stdout,
        )
    if args.cmd == "tokens" and args.tokens_cmd == "rotate":
        return cmd_tokens_rotate(
            args.role,
            grace=args.grace,
            secrets_root=(
                Path(args.secrets_root).expanduser()
                if args.secrets_root else None
            ),
            out=sys.stdout,
        )
    if args.cmd == "tokens" and args.tokens_cmd == "revoke":
        return cmd_tokens_revoke(
            args.token_id,
            secrets_root=(
                Path(args.secrets_root).expanduser()
                if args.secrets_root else None
            ),
            out=sys.stdout,
        )
    if args.cmd == "tokens" and args.tokens_cmd == "list":
        return cmd_tokens_list(
            secrets_root=(
                Path(args.secrets_root).expanduser()
                if args.secrets_root else None
            ),
            out=sys.stdout,
        )
    if args.cmd == "fairshare" and args.fairshare_cmd == "show":
        return cmd_fairshare_show(
            state_db=state_db,
            out=sys.stdout,
        )
    if args.cmd == "fairshare" and args.fairshare_cmd == "set":
        return cmd_fairshare_set(
            default_cap=args.default_cap,
            disable=args.disable,
            owner=args.owner,
            cap=args.cap,
            uncap=args.uncap,
            recap=args.recap,
            block=args.block,
            unblock=args.unblock,
            clear_owner=args.clear_owner,
            state_db=state_db,
            out=sys.stdout,
        )
    if args.cmd == "bootstrap":
        from xrlenv.cli.bootstrap import cmd_bootstrap
        return cmd_bootstrap(
            target=args.target,
            control_plane=args.control_plane,
            node_id=args.node_id,
            target_os=args.target_os,
            xrlenv_wheel=(
                Path(args.xrlenv_wheel).expanduser()
                if args.xrlenv_wheel else None
            ),
            xrlenv_repo=(
                Path(args.xrlenv_repo).expanduser()
                if args.xrlenv_repo else None
            ),
            xrlenv_version=args.xrlenv_version,
            runtime_user=args.runtime_user,
            install_root=(
                Path(args.install_root).expanduser()
                if args.install_root else None
            ),
            skip_operator_docker_group=args.skip_operator_docker_group,
            dry_run=args.dry_run,
            out=sys.stdout,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
