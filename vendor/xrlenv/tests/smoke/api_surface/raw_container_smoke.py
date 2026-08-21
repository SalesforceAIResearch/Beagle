"""P1.7.A.1 — End-to-end smoke for the raw-container session.

Exercises the full vertical: ``Client.acquire_container`` → control
plane picks a node → spec-21 ``AcquireContainerCommand`` → node-side
``RawContainerManager`` spawns a real docker container → exec a
command via ``ContainerExecCommand`` → destroy. No EnvAdapter, no
in-sandbox stub — the case-2/3 evaluation path.

Three modes (mirrors ``tests/smoke/cluster_smoke.py``):

1. **Embedded** (default, no ``--connect-host``): the script boots
   the control plane in this process. Convenient for laptop dev
   when no ``xrlenv up`` is already running. Stop any existing
   ``xrlenv up`` first; cloud nodes reconnect via systemd.

2. **Connect** (``--connect-host``, ``--consumer-token``): leaves
   the operator's existing ``xrlenv up`` running and dials it.
   The realistic shape (consumer + control plane on different
   machines).

3. **In-process LocalRuntime** (``--in-process``): the simplest
   topology — one Python process running everything. Useful for
   verifying the wire contract works without the gRPC hop.

Usage::

    # In-process (no grpc, single host) — fastest sanity check:
    python tests/smoke/raw_container_smoke.py --in-process

    # Embedded mode (replaces xrlenv up; cloud nodes reconnect):
    python tests/smoke/raw_container_smoke.py \\
        --grpc-port 50051 --min-nodes 1

    # Connect mode (leaves xrlenv up running; dials it):
    python tests/smoke/raw_container_smoke.py \\
        --connect-host 127.0.0.1 --connect-port 50051 \\
        --consumer-token "$XRLENV_CONSUMER_TOKEN"

Pre-req in non-in-process modes: at least one node-agent
attached AND that node has the smoke image (``busybox:latest``
by default, ``--image`` to override) locally available. Phase-1
contract: no implicit pull on the raw-container path.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from xrlenv import Client
from xrlenv.observability.logging import configure_logging

LOGGER = logging.getLogger("xrlenv.smoke.raw_container")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="raw_container_smoke",
        description=(
            "Run a single acquire/exec/destroy cycle against the "
            "raw-container session API. Tests the full vertical "
            "(SDK → control plane → node → docker → back) for "
            "case-2/3 evaluation harnesses."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--in-process", action="store_true",
        help="Use a LocalRuntime (no gRPC). Fastest sanity check.",
    )
    p.add_argument("--grpc-host", default="127.0.0.1")
    p.add_argument("--grpc-port", type=int, default=50051)
    p.add_argument(
        "--connect-host", default=None,
        help="Dial this host's existing xrlenv up instead of "
             "embedded mode.",
    )
    p.add_argument(
        "--connect-port", type=int, default=50051,
        help="Port for --connect-host.",
    )
    p.add_argument(
        "--consumer-token", default=None,
        help="Bearer token for --connect-host (or set "
             "$XRLENV_CONSUMER_TOKEN).",
    )
    p.add_argument(
        "--min-nodes", type=int, default=1,
        help="Wait for at least this many nodes to attach in "
             "embedded mode (default 1).",
    )
    p.add_argument(
        "--restart-grace", type=float, default=90.0,
        help="Seconds to wait for cloud VMs to reattach in "
             "embedded mode (default 90).",
    )
    p.add_argument(
        "--image", default="busybox:latest",
        help="Image to spawn the raw container from (default "
             "busybox:latest). Must already be present on the "
             "chosen node — no implicit pull.",
    )
    p.add_argument(
        "--cmd", nargs="+", default=["echo", "raw-container-smoke"],
        help="Command to exec inside the container (default: "
             "echo raw-container-smoke).",
    )
    p.add_argument(
        "--metrics-port", type=int, default=9090,
        help="Prometheus /metrics port for embedded mode.",
    )
    p.add_argument(
        "--admin-port", type=int, default=8080,
        help="Admin panel port for embedded mode.",
    )
    return p


async def _run_smoke(client: Client, *, image: str, cmd: list[str]) -> int:
    """Acquire → exec → destroy. Returns 0 on success, 1 on failure."""
    print(f"[smoke] acquiring container (image={image})", file=sys.stderr, flush=True)
    async with await client.acquire_container(
        image=image,
        command=["sleep", "infinity"],
    ) as session:
        print(
            f"[smoke] acquired rollout={session.rollout_id} "
            f"container={session.container_id[:12]} "
            f"node={session.node_id}",
            file=sys.stderr, flush=True,
        )
        result = await session.exec(cmd, timeout_s=10.0)
        print(
            f"[smoke] exec exit_code={result.exit_code} "
            f"timed_out={result.timed_out}",
            file=sys.stderr, flush=True,
        )
        if result.stdout:
            print(
                "[smoke] stdout:", result.stdout.decode("utf-8", "replace"),
                file=sys.stderr,
            )
        if result.stderr:
            print(
                "[smoke] stderr:", result.stderr.decode("utf-8", "replace"),
                file=sys.stderr,
            )
        if result.exit_code != 0 or result.timed_out:
            return 1
    print("[smoke] destroyed; SUCCESS", file=sys.stderr, flush=True)
    return 0


async def _run_in_process(args: argparse.Namespace) -> int:
    from xrlenv.control.runtime import build_local_runtime

    runtime = build_local_runtime(
        node_id="raw-smoke-local",
        run_dir_retention_days=None,
        # Skip the metrics server / admin panel — keep this fast.
        metrics_port=None,
    )
    await runtime.start()
    try:
        client = Client.in_process(runtime.service)
        return await _run_smoke(client, image=args.image, cmd=args.cmd)
    finally:
        await runtime.shutdown()


async def _run_embedded(args: argparse.Namespace) -> int:
    from xrlenv.control.distributed_runtime import build_distributed_runtime

    runtime = await build_distributed_runtime(
        grpc_host=args.grpc_host, grpc_port=args.grpc_port,
        metrics_port=args.metrics_port,
        admin_port=args.admin_port,
    )
    await runtime.start()
    try:
        # Wait for at least --min-nodes to attach.
        deadline = asyncio.get_event_loop().time() + args.restart_grace
        while True:
            attached = len(runtime.scheduler.nodes)
            if attached >= args.min_nodes:
                break
            if asyncio.get_event_loop().time() > deadline:
                print(
                    f"[smoke] FAIL: only {attached}/{args.min_nodes} "
                    f"nodes attached after {args.restart_grace}s",
                    file=sys.stderr,
                )
                return 1
            await asyncio.sleep(1.0)
        print(
            f"[smoke] {attached} node(s) attached: "
            f"{[n.node_id for n in runtime.scheduler.nodes]}",
            file=sys.stderr,
        )
        client = Client.in_process(runtime.service)
        return await _run_smoke(client, image=args.image, cmd=args.cmd)
    finally:
        await runtime.shutdown()


async def _run_connect(args: argparse.Namespace) -> int:
    import os
    token = args.consumer_token or os.environ.get("XRLENV_CONSUMER_TOKEN")
    client = Client.grpc(
        host=args.connect_host, port=args.connect_port, token=token,
    )
    try:
        return await _run_smoke(client, image=args.image, cmd=args.cmd)
    finally:
        await client.close()


async def main() -> int:
    args = _build_parser().parse_args()
    configure_logging()

    if args.in_process:
        return await _run_in_process(args)
    if args.connect_host is not None:
        return await _run_connect(args)
    return await _run_embedded(args)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
