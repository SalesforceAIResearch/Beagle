"""P1.7.B.2 W6: ``xrlenv images plan`` operator CLI handler.

Reads a ref list (file or repeated ``--ref``), dials the control
plane via the operator-token gRPC channel, invokes the
``PlanImageDistribution`` RPC, prints per-row results.

This is **operator-side**, NOT consumer-side. The audience's
docker-py drop-in harness is unchanged through this entire flow —
the CLI just populates ``StateStore.find_registered_preferred_home``
rows so subsequent ``client.containers.create(image=X)`` acquires
get steered to the planner-recorded home via the existing
``Scheduler.place(preferred_home_node=...)`` path.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import IO

import grpc

from xrlenv.api._pb2 import rollout_control_pb2 as rpb
from xrlenv.api._pb2 import rollout_control_pb2_grpc as rpb_grpc


def _parse_refs_file(path: Path, default_size: int) -> list[tuple[str, int]]:
    """Read one ref per line. Lines may carry an optional
    ``\\t<size_bytes>`` suffix (or space-separated). Lines starting
    with '#' are comments, blank lines skipped."""
    rows: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ref = parts[0]
            if len(parts) >= 2:
                try:
                    size = int(parts[1])
                except ValueError:
                    print(
                        f"[images plan] warning: bad size on line "
                        f"{raw!r}; using default {default_size}",
                        file=sys.stderr,
                    )
                    size = default_size
            else:
                size = default_size
            rows.append((ref, size))
    return rows


def cmd_images_plan(
    *,
    refs_file: Path | None,
    refs_inline: list[str],
    default_size_bytes: int,
    eager_prefetch: bool,
    control_host: str,
    control_port: int,
    operator_token: str | None,
    out: IO[str],
) -> int:
    """Dispatch the CLI. Returns 0 on success, 1 on partial failure
    (any row sealed status="failed"), 2 on input error."""
    if not refs_file and not refs_inline:
        print(
            "[images plan] error: pass --refs <file> or one or more "
            "--ref <X> (mutually exclusive).",
            file=sys.stderr,
        )
        return 2
    if refs_file and refs_inline:
        print(
            "[images plan] error: --refs and --ref are mutually "
            "exclusive.",
            file=sys.stderr,
        )
        return 2

    rows: list[tuple[str, int]] = []
    if refs_file is not None:
        rows = _parse_refs_file(refs_file, default_size_bytes)
    else:
        rows = [(ref, default_size_bytes) for ref in refs_inline]

    if not rows:
        print(
            "[images plan] error: ref list is empty.",
            file=sys.stderr,
        )
        return 2

    token = operator_token or os.environ.get("XRLENV_OPERATOR_TOKEN")
    if not token:
        print(
            "[images plan] error: --operator-token or "
            "$XRLENV_OPERATOR_TOKEN required.",
            file=sys.stderr,
        )
        return 2

    return asyncio.run(_run_plan(
        rows=rows,
        eager_prefetch=eager_prefetch,
        control_host=control_host,
        control_port=control_port,
        token=token,
        out=out,
    ))


async def _run_plan(
    *,
    rows: list[tuple[str, int]],
    eager_prefetch: bool,
    control_host: str,
    control_port: int,
    token: str,
    out: IO[str],
) -> int:
    target = f"{control_host}:{control_port}"
    print(
        f"[images plan] dialing {target}; "
        f"refs={len(rows)} eager_prefetch={eager_prefetch}",
        file=sys.stderr,
    )

    async with grpc.aio.insecure_channel(target) as channel:
        stub = rpb_grpc.RolloutControlStub(channel)
        req = rpb.PlanImageDistributionRequest(
            rows=[
                rpb.ImagePlanRow(
                    image_ref=ref,
                    size_bytes_hint=size,
                    replication=1,
                )
                for ref, size in rows
            ],
            eager_prefetch=eager_prefetch,
        )
        metadata = (("authorization", f"Bearer {token}"),)
        try:
            resp = await stub.PlanImageDistribution(
                req, metadata=metadata,
            )
        except grpc.aio.AioRpcError as exc:
            print(
                f"[images plan] error: RPC failed: {exc.code()} "
                f"{exc.details()}",
                file=sys.stderr,
            )
            return 1

    placed = 0
    deferred = 0
    failed = 0
    for r in resp.assignments:
        marker = {
            "placed": " ",
            "deferred": "?",
            "failed": "!",
        }.get(r.status, "?")
        home = r.preferred_home_node or "<no-home>"
        line = f"  [{marker}] {r.image_ref} → {home} ({r.status})"
        if r.error:
            line += f" — {r.error}"
        print(line, file=out)
        if r.status == "placed":
            placed += 1
        elif r.status == "deferred":
            deferred += 1
        elif r.status == "failed":
            failed += 1

    summary = (
        f"\n[images plan] placed={placed} deferred={deferred} "
        f"failed={failed} total={len(resp.assignments)}"
    )
    if eager_prefetch:
        summary += " (eager prefetch dispatched)"
    print(summary, file=out)
    return 1 if failed else 0
