"""P1.7.B end-to-end smoke for the docker-py drop-in (cluster mode).

Exercises the **full vertical** through the docker-py manager
surface — what swebench / harbor / any docker-py-using harness
would see when they swap ``docker.from_env`` for
``xrlenv.from_env(client=...)``:

    client = xrlenv.from_env(client=...)
    container = client.containers.run(image, command, detach=True)
    container.put_archive("/tmp", tar_bytes)
    result = container.exec_run(["echo", "hi"])
    container.remove(force=True)

Same three-mode structure as ``raw_container_smoke.py``:

1. **In-process LocalRuntime** (``--in-process``): single Python
   process, no gRPC. Fastest sanity check.
2. **Embedded** (default, no ``--connect-host``): script boots a
   control plane on ``--grpc-port``; cloud nodes reattach via
   systemd.
3. **Connect** (``--connect-host``, ``--consumer-token``): dials
   an already-running ``xrlenv up``.

Pre-req: at least one node-agent attached AND that node has
``--image`` (default ``busybox:latest``) locally available.
Phase-1 contract: no implicit pull on the raw-container path.

Distinct from ``raw_container_smoke.py``: that one uses the SDK
directly (``Client.acquire_container``); this one exercises the
docker-py drop-in's manager surface so the docker-API
translation layer (XrlenvAPIClient cluster mode) is itself
under test.

Usage::

    # In-process (no grpc, single host) — fastest sanity check:
    python tests/smoke/dropin_cluster_smoke.py --in-process

    # Embedded mode (replaces xrlenv up; cloud nodes reconnect):
    python tests/smoke/dropin_cluster_smoke.py \\
        --grpc-port 50051 --min-nodes 1

    # Connect mode (leaves xrlenv up running; dials it):
    python tests/smoke/dropin_cluster_smoke.py \\
        --connect-host 127.0.0.1 --connect-port 50051 \\
        --consumer-token "$XRLENV_CONSUMER_TOKEN"
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
import tarfile

import xrlenv
from xrlenv import Client
from xrlenv.observability.logging import configure_logging

LOGGER = logging.getLogger("xrlenv.smoke.dropin_cluster")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dropin_cluster_smoke",
        description=(
            "End-to-end smoke for the docker-py drop-in cluster mode. "
            "Exercises containers.run + put_archive + exec_run + "
            "remove via the docker-py manager surface so the "
            "XrlenvAPIClient cluster-mode translation layer is "
            "itself under test."
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
        help="Dial this host's existing xrlenv up.",
    )
    p.add_argument("--connect-port", type=int, default=50051)
    p.add_argument(
        "--consumer-token", default=None,
        help="Bearer token for --connect-host (or "
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
             "embedded mode.",
    )
    p.add_argument(
        "--image", default="busybox:latest",
        help="Image (must already be present on the chosen node).",
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


def _make_tar(name: str, content: bytes) -> bytes:
    """Build a single-file tar archive in-memory. swebench's
    harness builds these for ``put_archive``; we mirror the
    pattern."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _smoke_with_drop_in(
    docker_client: xrlenv.XrlenvDockerClient, *, image: str,
) -> int:
    """Run the sync smoke against a pre-built drop-in. Each entry
    point (``--in-process`` / embedded / ``--connect-host``) builds
    the drop-in differently to match the loop topology of its
    Client (see each ``_run_*`` for why)."""
    try:
        return _run_smoke_inner(docker_client, image=image)
    finally:
        docker_client.close()


def _run_smoke_inner(
    docker_client: xrlenv.XrlenvDockerClient, *, image: str,
) -> int:
    """Same body as the original ``_run_smoke`` but receives a
    pre-built drop-in (so the construction strategy can vary per
    mode)."""
    print(
        f"[smoke] client.containers.create(image={image!r}, "
        f"command=['sleep', 'infinity'])",
        file=sys.stderr, flush=True,
    )
    container = docker_client.containers.create(
        image, command=["sleep", "infinity"],
        labels={"xrlenv.smoke": "dropin-cluster"},
    )
    print(
        f"[smoke] container created id={container.id[:12]} "
        f"(via cluster Client.acquire_container)",
        file=sys.stderr, flush=True,
    )
    container.start()  # no-op in cluster mode
    print("[smoke] container.start() returned (no-op)", file=sys.stderr)

    try:
        tarball = _make_tar("hello.txt", b"hello from dropin smoke\n")
        ok = container.put_archive("/tmp", tarball)
        print(
            f"[smoke] container.put_archive(/tmp, hello.txt) -> {ok}",
            file=sys.stderr,
        )
        exit_code, out = container.exec_run(
            ["cat", "/tmp/hello.txt"], demux=True,
        )
        if isinstance(out, tuple):
            stdout_bytes, _stderr_bytes = out
        else:
            stdout_bytes = out or b""
        print(
            f"[smoke] container.exec_run(cat /tmp/hello.txt) "
            f"exit_code={exit_code} stdout="
            f"{stdout_bytes.decode('utf-8', 'replace')!r}",
            file=sys.stderr,
        )
        if exit_code != 0:
            print("[smoke] FAIL: non-zero exit", file=sys.stderr)
            return 1
        if b"hello from dropin smoke" not in stdout_bytes:
            print(
                f"[smoke] FAIL: stdout doesn't carry the file's "
                f"bytes; got {stdout_bytes!r}",
                file=sys.stderr,
            )
            return 1
        chunks_iter, _stat = container.get_archive("/tmp/hello.txt")
        tar_bytes = b"".join(chunks_iter)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
            roundtripped = tf.extractfile("hello.txt").read()  # type: ignore[union-attr]
        print(
            f"[smoke] container.get_archive(/tmp/hello.txt) "
            f"round-tripped {len(roundtripped)} bytes",
            file=sys.stderr,
        )
        if roundtripped != b"hello from dropin smoke\n":
            print(
                "[smoke] FAIL: round-tripped bytes don't match",
                file=sys.stderr,
            )
            return 1
        exit_code, output_iter = container.exec_run(
            ["sh", "-c", "echo first; echo second; echo third"],
            stream=True,
        )
        chunks = list(output_iter)
        combined = b"".join(chunks)
        print(
            f"[smoke] streaming exec_run yielded {len(chunks)} "
            f"chunk(s); combined bytes={combined!r}",
            file=sys.stderr,
        )
        if (
            b"first" not in combined
            or b"second" not in combined
            or b"third" not in combined
        ):
            print(
                "[smoke] FAIL: streaming exec missing expected bytes",
                file=sys.stderr,
            )
            return 1
    finally:
        print(
            "[smoke] container.remove(force=True) -> destroy via SDK",
            file=sys.stderr,
        )
        container.remove(force=True)

    print("[smoke] SUCCESS", file=sys.stderr, flush=True)
    return 0


def _run_smoke(client: Client, *, image: str) -> int:
    """Legacy entry point kept for backwards-compatible callers
    (e.g. ``--in-process``). Builds the drop-in around an already-
    constructed Client without runner-backed dispatch — fine when
    the Client carries no loop-bound state (LocalRuntime in-process
    is the working case)."""
    print(
        "[smoke] building xrlenv.from_env drop-in (cluster mode) "
        "around the SDK Client",
        file=sys.stderr, flush=True,
    )
    docker_client = xrlenv.from_env(client=client)
    return _smoke_with_drop_in(docker_client, image=image)


async def _run_in_process(args: argparse.Namespace) -> int:
    from xrlenv.control.runtime import build_local_runtime

    runtime = build_local_runtime(
        node_id="dropin-smoke-local",
        run_dir_retention_days=None,
        metrics_port=None,
    )
    await runtime.start()
    try:
        client = Client.in_process(runtime.service)
        # LocalRuntime: no gRPC, no remote-node loop-bound state;
        # the legacy fresh-loop drop-in is fine here.
        return await asyncio.to_thread(
            _run_smoke, client, image=args.image,
        )
    finally:
        await runtime.shutdown()


async def _run_embedded(args: argparse.Namespace) -> int:
    from xrlenv.compat.docker_client import _DropInRunner
    from xrlenv.control.distributed_runtime import build_distributed_runtime

    runtime = await build_distributed_runtime(
        grpc_host=args.grpc_host, grpc_port=args.grpc_port,
        metrics_port=args.metrics_port,
        admin_port=args.admin_port,
    )
    await runtime.start()
    try:
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
        # Embedded runtime: ``RemoteNodeTransport`` carries
        # asyncio.Queue / asyncio.Future bound to THIS loop. Build
        # an attached runner so the worker thread's drop-in calls
        # dispatch via ``run_coroutine_threadsafe`` back to this
        # loop — no fresh-loop / cross-loop hazards.
        runner = _DropInRunner(loop=asyncio.get_running_loop())
        docker_client = xrlenv.from_env(client=client, runner=runner)
        return await asyncio.to_thread(
            _smoke_with_drop_in, docker_client, image=args.image,
        )
    finally:
        await runtime.shutdown()


def _run_connect(args: argparse.Namespace) -> int:
    """Connect mode is fully sync: the self-contained
    ``from_env(grpc_host=...)`` factory owns its runner + Client.
    No outer asyncio loop needed; in particular we DON'T wrap
    this in ``asyncio.to_thread`` from a parent loop because that
    pattern is exactly what the cross-loop bug stems from."""
    import os
    token = args.consumer_token or os.environ.get("XRLENV_CONSUMER_TOKEN")
    docker_client = xrlenv.from_env(
        grpc_host=args.connect_host,
        grpc_port=args.connect_port,
        consumer_token=token,
    )
    return _smoke_with_drop_in(docker_client, image=args.image)


async def _async_main(args: argparse.Namespace) -> int:
    if args.in_process:
        return await _run_in_process(args)
    return await _run_embedded(args)


def main() -> int:
    args = _build_parser().parse_args()
    configure_logging()
    if args.connect_host is not None:
        # Connect mode: fully sync, runner owns its own loop.
        # Do NOT wrap in asyncio.run — that would force the
        # drop-in's runner-loop to coexist with an outer loop in
        # the same process, recreating the cross-loop hazard if
        # someone later adds an inner await.
        return _run_connect(args)
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())
