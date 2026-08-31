"""Cluster smoke — drive a live ``xrlenv up`` control plane with real
cloud VMs (or local node agents) as the data plane.

Two modes of operation:

1. **Embedded** (default, no ``--connect-host``): the script boots the
   control plane in this process, on the same port the operator's SSH
   reverse tunnels already point at (default 50051). The operator stops
   any existing ``xrlenv up`` before running; cloud VMs reconnect via
   their systemd restart loop. Useful when you don't yet have a
   long-running ``xrlenv up`` you want to keep.

2. **Connect** (``--connect-host``, ``--consumer-token``): the script
   leaves the operator's existing ``xrlenv up`` alone and dials it via
   ``Client.grpc(host, port, token)``. This is the topology the
   spec-05 SDK is intended for — control plane on one host, trainer
   process on another (or the same host but a separate Python
   process).

Both modes wait for ``--min-nodes`` nodes to attach, then submit
``--rollouts`` hello-shell rollouts and print one-line summaries.

Usage::

    # Embedded mode (replaces `xrlenv up` for the smoke):
    python tests/smoke/cluster_bringup/cluster_smoke.py \\
        --grpc-port 50051 --min-nodes 2 --rollouts 4 --spread

    # Connect mode (leaves `xrlenv up` running, dials it):
    python tests/smoke/cluster_bringup/cluster_smoke.py \\
        --connect-host 127.0.0.1 --connect-port 50051 \\
        --consumer-token "$XRLENV_CONSUMER_TOKEN" \\
        --min-nodes 2 --rollouts 4 --spread

Environment:

- ``XRLENV_SECRETS_ROOT`` — override the default ``~/.xrlenv/secrets/``.
- ``XRLENV_CONSUMER_TOKEN`` — alternative to ``--consumer-token``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pprint
import sys
from pathlib import Path

from xrlenv import Client
from xrlenv.control import build_distributed_runtime
from xrlenv.observability.logging import configure_logging

LOGGER = logging.getLogger("xrlenv.smoke.cluster")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cluster_smoke",
        description=(
            "Run a multi-rollout acceptance smoke against a live control "
            "plane bound on a port your existing SSH tunnels already point "
            "at. Replaces `xrlenv up` for the duration of the run."
        ),
    )
    p.add_argument(
        "--grpc-host", default="127.0.0.1",
        help="Bind address for the gRPC server (default 127.0.0.1, matching "
             "the typical SSH reverse-tunnel topology).",
    )
    p.add_argument(
        "--grpc-port", type=int, default=50051,
        help="Bind port for the gRPC server. Use the same port your SSH "
             "reverse tunnels forward to (default 50051).",
    )
    p.add_argument(
        "--admin-port", type=int, default=8080,
        help="Admin panel port (default 8080).",
    )
    p.add_argument(
        "--metrics-port", type=int, default=9090,
        help="Prometheus /metrics port (default 9090).",
    )
    p.add_argument(
        "--runs-root", type=Path,
        default=Path.home() / ".xrlenv" / "runs",
        help="Per-rollout artifact root (default ~/.xrlenv/runs).",
    )
    p.add_argument(
        "--state-db", type=Path,
        default=Path.home() / ".xrlenv" / "state.db",
        help="State database path (default ~/.xrlenv/state.db).",
    )
    p.add_argument(
        "--template", default="hello-shell",
        help="Template name to roll out (default hello-shell).",
    )
    p.add_argument(
        "--rollouts", type=int, default=4,
        help="Number of rollouts to submit (default 4).",
    )
    p.add_argument(
        "--max-steps", type=int, default=3,
        help="Steps per rollout (default 3).",
    )
    p.add_argument(
        "--min-nodes", type=int, default=2,
        help="Number of nodes that must attach before submitting rollouts "
             "(default 2 — one GCP, one AWS).",
    )
    p.add_argument(
        "--restart-grace", type=float, default=90.0,
        help="Seconds to wait for cloud VMs to reattach (default 90). "
             "The xrlenv-node grpc_link reconnect backoff caps at 30s, so a "
             "node that's been failing for a while can need up to ~30-60s "
             "to land its next attempt after the smoke starts listening. "
             "Workaround if you want faster turnaround: SSH to each VM and "
             "run `sudo systemctl restart xrlenv-node` to reset the backoff "
             "to 1s.",
    )
    p.add_argument(
        "--continue-on-partial", action="store_true",
        help="Don't abort if fewer than --min-nodes attach — run the smoke "
             "against whatever connected. Useful when debugging a single "
             "VM's auth/network setup.",
    )
    p.add_argument(
        "--spread", action="store_true",
        help="Force per-node distribution by tagging every rollout with the "
             "same task_key. In **embedded** mode the smoke also lowers the "
             "scheduler's max_runs_per_task to ceil(rollouts/nodes). In "
             "**connect** mode the server's max_runs_per_task is fixed by "
             "however the operator launched `xrlenv up` — for deterministic "
             "spread, launch with `xrlenv up --max-runs-per-task <N>` where "
             "N=ceil(rollouts/nodes). Without that, the capacity-aware "
             "scheduler may legitimately stack all rollouts on the largest "
             "node when hello-shell-sized jobs fit comfortably. Use --spread "
             "to *prove* every attached VM is healthy in an acceptance run.",
    )
    p.add_argument(
        "--connect-host", default=None,
        help="Connect to an existing `xrlenv up` instead of booting an "
             "embedded runtime. When set, the script uses Client.grpc(host, "
             "port, token=...) for the smoke; the operator's `xrlenv up` "
             "stays untouched.",
    )
    p.add_argument(
        "--connect-port", type=int, default=50051,
        help="gRPC port on --connect-host (default 50051).",
    )
    p.add_argument(
        "--consumer-token", default=None,
        help="Bearer token for --connect mode. Falls back to the "
             "XRLENV_CONSUMER_TOKEN env var. Required when --connect-host "
             "is set against an auth-enabled control plane.",
    )
    return p


async def _wait_for_nodes(runtime: object, *, min_nodes: int, timeout_s: float) -> list[str]:
    """Poll the live registry until ``min_nodes`` are attached or we time out."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        ids = sorted(n.node_id for n in runtime.scheduler.nodes)  # type: ignore[attr-defined]
        if len(ids) >= min_nodes:
            LOGGER.info("scheduler has %d node(s) attached: %s", len(ids), ids)
            return ids
        await asyncio.sleep(1.0)
    ids_now = sorted(n.node_id for n in runtime.scheduler.nodes)  # type: ignore[attr-defined]
    raise TimeoutError(
        f"only {len(ids_now)} of the required {min_nodes} node(s) attached "
        f"within {timeout_s:.0f}s: {ids_now or '(none)'}"
    )


async def _run_connect_mode(args: argparse.Namespace) -> int:
    """Drive a smoke against an existing ``xrlenv up`` via Client.grpc.

    The remote control plane already has its NodeRegistry; we don't
    have an in-process scheduler to inspect to confirm node count.
    We dispatch rollouts directly and trust the server's placement /
    capacity-exhausted response.
    """
    import os

    token = args.consumer_token or os.environ.get("XRLENV_CONSUMER_TOKEN")
    LOGGER.info(
        "Connect mode: dialing %s:%d via Client.grpc (consumer token %s)",
        args.connect_host, args.connect_port,
        "set" if token else "NOT SET (server may reject)",
    )
    client = Client.grpc(args.connect_host, args.connect_port, token=token)
    finished, truncated, failed = await _drive_rollouts(client, args)
    return _report_and_exit(finished, truncated, failed, args.rollouts)


async def main() -> int:
    args = _build_parser().parse_args()
    configure_logging(level=logging.INFO, log_format="auto")

    if args.connect_host is not None:
        return await _run_connect_mode(args)

    LOGGER.info(
        "Embedded mode: gRPC=%s:%d state_db=%s runs_root=%s",
        args.grpc_host, args.grpc_port, args.state_db, args.runs_root,
    )
    LOGGER.info(
        "If the bind fails with `address already in use`, stop the "
        "existing `xrlenv up` first or use --connect-host to dial it.",
    )

    # When --spread is on we cap max_runs_per_task at ceil(rollouts /
    # min_nodes), which forces the scheduler to spill once that many
    # same-task_key rollouts are stacked on a node. Default is 4 (matches
    # phase-0 GRPO group size) — keep it the production default otherwise.
    import math
    scheduler_max_runs_per_task = (
        max(1, math.ceil(args.rollouts / max(1, args.min_nodes)))
        if args.spread else 4
    )

    runtime = await build_distributed_runtime(
        grpc_host=args.grpc_host,
        grpc_port=args.grpc_port,
        runs_root=args.runs_root,
        state_db_path=args.state_db,
        admin_host="127.0.0.1",
        admin_port=args.admin_port,
        metrics_host="127.0.0.1",
        metrics_port=args.metrics_port,
        scheduler_max_runs_per_task=scheduler_max_runs_per_task,
    )
    await runtime.start()

    try:
        try:
            await _wait_for_nodes(
                runtime, min_nodes=args.min_nodes, timeout_s=args.restart_grace,
            )
        except TimeoutError as exc:
            attached = sorted(n.node_id for n in runtime.scheduler.nodes)  # type: ignore[attr-defined]
            if args.continue_on_partial and attached:
                LOGGER.warning(
                    "%s — proceeding with %d node(s) because --continue-on-partial is set: %s",
                    exc, len(attached), attached,
                )
            else:
                LOGGER.error("%s", exc)
                LOGGER.error(
                    "Check on each VM: `sudo journalctl -u xrlenv-node -n 30 --no-pager`. "
                    "Pro tip: `sudo systemctl restart xrlenv-node` resets the gRPC "
                    "reconnect backoff to 1s, so the next attempt fires immediately.",
                )
                return 2

        client = Client.in_process(runtime.service)
        finished, truncated, failed = await _drive_rollouts(client, args)
        return _report_and_exit(finished, truncated, failed, args.rollouts)
    finally:
        await runtime.shutdown()


async def _drive_rollouts(
    client: Client,
    args: argparse.Namespace,
) -> tuple[list, list, list[tuple[str | None, str, str]]]:
    """Submit ``args.rollouts`` rollouts via the given client.

    Same shape regardless of whether ``client`` is in-process or
    backed by ``Client.grpc(...)``. Starts are serialised so the
    scheduler's ``max_runs_per_task`` cap (when ``--spread``) sees
    each previous placement recorded; drives run concurrently via
    ``asyncio.gather``.
    """
    finished: list = []
    truncated: list = []
    failed: list[tuple[str | None, str, str]] = []
    try:
        inits = [
            {"max_steps": args.max_steps, "cwd": "/sandbox", "rollout_idx": i}
            for i in range(args.rollouts)
        ]

        async def _hello_shell_policy() -> dict[str, str]:
            return {"cmd": "echo cluster-smoke"}

        task_keys = (
            ["cluster-smoke-spread"] * args.rollouts if args.spread else None
        )
        sessions = []
        for i, init_payload in enumerate(inits):
            tk = task_keys[i] if task_keys else None
            try:
                s = await client.rollout(
                    template=args.template, init=init_payload, task_key=tk,
                )
                sessions.append(s)
            except Exception as exc:
                failed.append((None, "start_failed", str(exc)))
                LOGGER.error("rollout %d start failed: %s", i, exc)

        async def _drive_session(session: object) -> None:
            async with session:  # type: ignore[attr-defined]
                try:
                    while not session.done:  # type: ignore[attr-defined]
                        await session.step(await _hello_shell_policy())  # type: ignore[attr-defined]
                except Exception as exc:
                    failed.append(
                        (session.rollout_id, "drive_failed", str(exc)),  # type: ignore[attr-defined]
                    )
                    return
            traj = session.trajectory  # type: ignore[attr-defined]
            if traj.status.value == "truncated":
                truncated.append(traj)
            else:
                finished.append(traj)

        await asyncio.gather(
            *[_drive_session(s) for s in sessions],
            return_exceptions=True,
        )
    finally:
        await client.close()
    return finished, truncated, failed


def _report_and_exit(
    finished: list,
    truncated: list,
    failed: list[tuple[str | None, str, str]],
    expected_total: int,
) -> int:
    print("\n=== Scenario-1 acceptance results ===")
    for traj in finished + truncated:
        pprint.pp({
            "rollout_id": traj.rollout_id,
            "status": traj.status.value,
            "steps": len(traj.steps),
            "node": traj.metadata.get("node_id"),
            "final_reward": traj.metadata.get("final_reward"),
        })
    for rollout_id, reason, message in failed:
        pprint.pp({
            "rollout_id": rollout_id,
            "status": "failed",
            "reason": reason,
            "error": message,
        })

    nodes_used = sorted({
        t.metadata.get("node_id") for t in finished
        if t.metadata.get("node_id")
    })
    LOGGER.info(
        "%d / %d rollouts sealed as finished across %d node(s): %s",
        len(finished), expected_total, len(nodes_used), nodes_used,
    )
    return 0 if len(finished) == expected_total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
