"""spec 07 — live smoke for the running-container egress restriction.

Proves on a REAL node what the unit tests can only fake: that
``ContainerSession.apply_egress`` actually installs an iptables OUTPUT chain
into the container's netns (via ``nsenter`` on the node), so a restricted
container can reach ONLY the declared allowlist (+ loopback) while cloud
metadata, an arbitrary external IP, and (with a tighter list) the open
internet are blocked.

The cycle mirrors the intended open-setup→tighten flow:

  1. acquire a container on the OPEN bridge
  2. install curl over the open network (bootstrap phase)
  3. BEFORE: confirm the gateway AND an arbitrary external IP are reachable
     (proves the network is genuinely open pre-restriction)
  4. resolve the gateway host -> IPv4s (host-side), pin them in the
     container's /etc/hosts (so the client needs no DNS once egress is cut)
  5. ``apply_egress`` the gateway /32s on the gateway port only
  6. AFTER: gateway still reachable; external IP + cloud metadata now BLOCKED
  7. (optional) re-tighten to an EMPTY allowlist and confirm even the
     gateway is now blocked (block-all)
  8. destroy

PASS requires the BEFORE→AFTER contrast: open before, gateway-only after.

Modes mirror ``raw_container_smoke.py``. The realistic one for a dev cluster
is **connect** — dial an existing ``xrlenv up``:

    python tests/smoke/api_surface/egress_restriction_smoke.py \\
        --connect-host <dev-control-plane> --connect-port 50051 \\
        --consumer-token "$XRLENV_CONSUMER_TOKEN" \\
        --gateway-url "$SFR_GATEWAY_OPENAI_URL"

Pre-reqs (non-in-process): a node attached whose host has ``nsenter`` +
``iptables`` + ``ip6tables`` and the node process holds CAP_NET_ADMIN, and
the ``--image`` is present on that node (no implicit pull on the raw path).
``--image`` must have a package manager for the curl install (alpine/apk or
debian/apt) OR already ship curl (pass ``--no-install-curl``).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
from urllib.parse import urlparse

from xrlenv import Client
from xrlenv.backends.egress import EgressAllowlist, EgressRule
from xrlenv.observability.logging import configure_logging

_DEFAULT_GATEWAY = "https://gateway.salesforceresearch.ai/openai/process/v1/"
# A stable external IP used as the "open internet" probe target. Cloudflare
# 1.1.1.1 answers on 443; reaching it by IP needs no DNS, so the probe tests
# the iptables REJECT directly (not DNS).
_EXTERNAL_PROBE_IP = "1.1.1.1"
_METADATA_IP = "169.254.169.254"

# Portable curl bootstrap (alpine/apk or debian/apt), mirroring monet's install.
_INSTALL_CURL = (
    "set -e\n"
    "if command -v curl >/dev/null 2>&1; then exit 0; fi\n"
    "if [ -f /etc/alpine-release ]; then\n"
    "  apk add --no-cache curl ca-certificates\n"
    "else\n"
    "  apt-get update -qq && apt-get install -y --no-install-recommends "
    "curl ca-certificates\n"
    "fi\n"
)


def _print(msg: str) -> None:
    print(f"[egress-smoke] {msg}", file=sys.stderr, flush=True)


async def _exec(session, cmd: str, *, user: str | None = None, timeout_s: float = 60.0):
    return await session.exec(["sh", "-c", cmd], user=user, timeout_s=timeout_s)


async def _curl_reachable(session, target: str, *, max_time: int = 8) -> tuple[bool, str]:
    """True if curl connects to ``target`` (any HTTP response, even 401/403).

    ``-o /dev/null`` discards the body; exit 0 = a response arrived (reachable),
    non-zero = blocked / refused / timed out. Returns (reachable, detail).
    """
    res = await _exec(
        session,
        f'curl -sS -o /dev/null -w "%{{http_code}}" --max-time {max_time} '
        f"{target} 2>/dev/null; echo \" rc=$?\"",
        timeout_s=max_time + 15.0,
    )
    out = (res.stdout or b"").decode("utf-8", "replace").strip()
    reachable = " rc=0" in out
    return reachable, out


async def _run_smoke(
    client: Client, *, image: str, gateway_url: str, install_curl: bool,
    check_block_all: bool,
) -> int:
    host = urlparse(gateway_url).hostname
    if not host:
        _print(f"FAIL: could not parse host from --gateway-url {gateway_url!r}")
        return 1
    port = urlparse(gateway_url).port or 443
    infos = socket.getaddrinfo(host, port, family=socket.AF_INET, proto=socket.IPPROTO_TCP)
    gw_ips = sorted({ai[4][0] for ai in infos})
    if not gw_ips:
        _print(f"FAIL: no IPv4 for gateway host {host!r}")
        return 1
    _print(f"gateway {host}:{port} -> {gw_ips}")

    failures: list[str] = []
    _print(f"acquiring container (image={image}) on the open bridge")
    async with await client.acquire_container(
        image=image, command=["sleep", "infinity"],
    ) as session:
        _print(
            f"acquired rollout={session.rollout_id} "
            f"container={session.container_id[:12]} node={session.node_id}",
        )

        if install_curl:
            _print("installing curl (open network)…")
            r = await _exec(session, _INSTALL_CURL, user="root", timeout_s=300.0)
            if r.exit_code != 0:
                _print(f"FAIL: curl install rc={r.exit_code} "
                       f"stderr={(r.stderr or b'').decode('utf-8', 'replace')[:400]}")
                return 1

        # ── BEFORE: the network is genuinely open ────────────────────────────
        gw_ok, gw_d = await _curl_reachable(session, gateway_url)
        ext_ok, ext_d = await _curl_reachable(session, f"https://{_EXTERNAL_PROBE_IP}")
        _print(f"BEFORE  gateway reachable={gw_ok} ({gw_d}) ; "
               f"external {_EXTERNAL_PROBE_IP} reachable={ext_ok} ({ext_d})")
        if not gw_ok:
            failures.append("BEFORE: gateway should be reachable on the open bridge")
        if not ext_ok:
            failures.append(f"BEFORE: {_EXTERNAL_PROBE_IP} should be reachable (open)")

        # ── pin /etc/hosts so the gateway resolves with no DNS, then tighten ──
        pin = "; ".join(
            f"printf '%s %s\\n' {ip} {host} >> /etc/hosts" for ip in gw_ips
        )
        await _exec(session, pin, user="root", timeout_s=30.0)
        allowlist = EgressAllowlist(
            rules=tuple(EgressRule(cidr=f"{ip}/32", ports=(port,)) for ip in gw_ips),
        )
        _print(f"apply_egress -> {[r.cidr for r in allowlist.rules]} port={port}")
        await session.apply_egress(allowlist)

        # ── AFTER: gateway only ──────────────────────────────────────────────
        gw_ok2, gw_d2 = await _curl_reachable(session, gateway_url)
        ext_ok2, ext_d2 = await _curl_reachable(session, f"https://{_EXTERNAL_PROBE_IP}", max_time=6)
        md_ok, md_d = await _curl_reachable(session, f"http://{_METADATA_IP}/latest/meta-data/", max_time=6)
        _print(f"AFTER   gateway reachable={gw_ok2} ({gw_d2})")
        _print(f"AFTER   external {_EXTERNAL_PROBE_IP} reachable={ext_ok2} ({ext_d2}) "
               f"[expect blocked]")
        _print(f"AFTER   metadata {_METADATA_IP} reachable={md_ok} ({md_d}) "
               f"[expect blocked]")
        if not gw_ok2:
            failures.append("AFTER: gateway should STILL be reachable (allowlisted)")
        if ext_ok2:
            failures.append(f"AFTER: {_EXTERNAL_PROBE_IP} should be BLOCKED")
        if md_ok:
            failures.append(f"AFTER: metadata {_METADATA_IP} should be BLOCKED")

        # ── optional: re-tighten to block-all and confirm gateway blocked ────
        if check_block_all:
            _print("apply_egress -> EMPTY (block-all)")
            await session.apply_egress(EgressAllowlist())
            gw_ok3, gw_d3 = await _curl_reachable(session, gateway_url, max_time=6)
            _print(f"BLOCK-ALL gateway reachable={gw_ok3} ({gw_d3}) [expect blocked]")
            if gw_ok3:
                failures.append("BLOCK-ALL: gateway should be BLOCKED under empty allowlist")

    if failures:
        _print("RESULT: FAIL")
        for f in failures:
            _print(f"  - {f}")
        return 1
    _print("RESULT: PASS — gateway-only egress enforced on a live node")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="egress_restriction_smoke",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--connect-host", default=None,
                   help="Dial this host's existing xrlenv up (the realistic mode).")
    p.add_argument("--connect-port", type=int, default=50051)
    p.add_argument("--consumer-token", default=None,
                   help="Bearer token (or set $XRLENV_CONSUMER_TOKEN).")
    p.add_argument("--in-process", action="store_true",
                   help="Run everything in one process via a LocalRuntime.")
    p.add_argument("--grpc-host", default="127.0.0.1")
    p.add_argument("--grpc-port", type=int, default=50051)
    p.add_argument("--metrics-port", type=int, default=9090)
    p.add_argument("--admin-port", type=int, default=8080)
    p.add_argument("--min-nodes", type=int, default=1)
    p.add_argument("--restart-grace", type=float, default=90.0)
    p.add_argument("--image", default="alpine:latest",
                   help="Container image (must be present on the node).")
    p.add_argument("--gateway-url", default=os.environ.get("SFR_GATEWAY_OPENAI_URL") or _DEFAULT_GATEWAY,
                   help="LLM gateway URL to allowlist (default $SFR_GATEWAY_OPENAI_URL).")
    p.add_argument("--no-install-curl", dest="install_curl", action="store_false",
                   help="Skip the curl bootstrap (image already ships curl).")
    p.add_argument("--check-block-all", action="store_true",
                   help="Also verify an empty allowlist blocks even the gateway.")
    return p


async def main() -> int:
    args = _build_parser().parse_args()
    configure_logging()
    kw = dict(
        image=args.image, gateway_url=args.gateway_url,
        install_curl=args.install_curl, check_block_all=args.check_block_all,
    )
    if args.in_process:
        from xrlenv.control.runtime import build_local_runtime
        runtime = build_local_runtime(node_id="egress-smoke-local", metrics_port=None)
        await runtime.start()
        try:
            return await _run_smoke(Client.in_process(runtime.service), **kw)
        finally:
            await runtime.shutdown()
    if args.connect_host is not None:
        token = args.consumer_token or os.environ.get("XRLENV_CONSUMER_TOKEN")
        client = Client.grpc(host=args.connect_host, port=args.connect_port, token=token)
        try:
            return await _run_smoke(client, **kw)
        finally:
            await client.close()
    # embedded
    from xrlenv.control.distributed_runtime import build_distributed_runtime
    runtime = await build_distributed_runtime(
        grpc_host=args.grpc_host, grpc_port=args.grpc_port,
        metrics_port=args.metrics_port, admin_port=args.admin_port,
    )
    await runtime.start()
    try:
        deadline = asyncio.get_event_loop().time() + args.restart_grace
        while len(runtime.scheduler.nodes) < args.min_nodes:
            if asyncio.get_event_loop().time() > deadline:
                _print(f"FAIL: <{args.min_nodes} nodes attached")
                return 1
            await asyncio.sleep(1.0)
        return await _run_smoke(Client.in_process(runtime.service), **kw)
    finally:
        await runtime.shutdown()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
