#!/usr/bin/env python3
"""Agent-agnostic LLM Gateway Express proxy — reach the gateway from a cluster node.

Cluster/HyperPod nodes usually can't reach LLM Gateway Express directly, but your
laptop can. This reproduces monet's `express-proxy` as a **standalone, agent-agnostic**
tool (stdlib only — no monet, no beagle): a streaming HTTP relay on the laptop + an
SSH reverse tunnel to the node, so *any* agent that speaks the gateway's
OpenAI-compatible API through `LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL` works, with no
change to the agent.

Data path:

    cluster agent  →  http://127.0.0.1:<remote-port>/chat/completions
                   →  ssh -R reverse tunnel  →  laptop relay (127.0.0.1:<local-port>)
                   →  https://<gateway>/chat/completions

The relay is required (not a bare `ssh -R`): the upstream is HTTPS with a specific
hostname, so the relay terminates TLS + rewrites Host, and it back-stops key
injection (round-robin over LLM_GATEWAY_EXPRESS_API_KEY_LIST for requests with no
Authorization header). It serves exactly the gateway's two paths — `/chat/completions`
and `/responses` (no `/v1` prefix) — and streams SSE responses through.

Usage
-----
On your **laptop** (which can reach the gateway):

    export LLM_GATEWAY_EXPRESS_API_KEY_LIST="key-1,key-2,key-3"   # or _API_KEY
    python3 scripts/gateway/gateway_proxy.py serve --remote my-remote
    # or the wrapper:  scripts/gateway/laptop.sh my-remote

`--remote` is any `ssh` target / config alias. It starts the relay, opens the tunnel
(auto-reconnecting, same URL preserved), and prints the env to set on the cluster.

On the **cluster** node, verify + wire it (the agent forwards these via its model env):

    export LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL=http://127.0.0.1:18080/
    export LLM_GATEWAY_EXPRESS_API_KEY_LIST="key-1,key-2,key-3"   # same list, so the agent sends its own key
    python3 scripts/gateway/gateway_proxy.py check                # confirms the tunnel + relay are reachable

In beagle, that maps to the run config's `model` block: `provider:
llm-gateway-express-local-proxy`, `env: [LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL,
LLM_GATEWAY_EXPRESS_API_KEY_LIST]` — no agent code changes.
"""

from __future__ import annotations

import argparse
import http.client
import itertools
import json
import re
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# The gateway monet's `express` provider targets by default (overridable).
DEFAULT_UPSTREAM = "https://eng-ai-model-gateway.sfproxy.devx-preprod.aws-esvc1-useast2.aws.sfdc.cl/"
DEFAULT_PORT = 18080
ALLOWED_PATHS = frozenset({"/chat/completions", "/responses"})  # gateway's OpenAI paths, no /v1
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
})


# --- pure helpers (unit-tested) ----------------------------------------------


def parse_keys(raw: str | None) -> list[str]:
    """Split an LLM_GATEWAY_EXPRESS_API_KEY_LIST (comma/semicolon/whitespace) → keys,
    de-duped, order-preserving."""
    if not raw:
        return []
    seen: dict[str, None] = {}
    for k in re.split(r"[\s,;]+", raw.strip()):
        if k:
            seen.setdefault(k, None)
    return list(seen)


class KeyPool:
    """Thread-safe round-robin over the key list (the relay's fallback for keyless
    requests). Empty pool → ``next()`` returns ``None``."""

    def __init__(self, keys: list[str]) -> None:
        self._keys = keys
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def next(self) -> str | None:
        if not self._keys:
            return None
        with self._lock:
            return self._keys[next(self._counter) % len(self._keys)]


def split_upstream(url: str) -> tuple[str, int, str]:
    """``https://host[:port]/base/`` → ``(host, port, base_path)`` (base_path has no
    trailing slash; port defaults to 443)."""
    u = urlsplit(url)
    if u.scheme != "https":
        raise ValueError(f"upstream must be https://, got {url!r}")
    return u.hostname or "", (u.port or 443), u.path.rstrip("/")


def forward_headers(incoming: dict[str, str], *, authorization: str, host: str) -> dict[str, str]:
    """Copy request headers minus hop-by-hop / host / content-length / any incoming
    Authorization, then set our own Authorization + Host (no dupes, any case)."""
    drop = _HOP_BY_HOP | {"authorization"}
    out = {k: v for k, v in incoming.items() if k.lower() not in drop}
    out["Authorization"] = authorization
    out["Host"] = host
    return out


def hostport(s: str, default_host: str = "0.0.0.0") -> tuple[str, int]:
    """``"host:port"`` / ``":port"`` / ``"port"`` → ``(host, port)``."""
    s = s.strip()
    if ":" in s:
        host, _, port = s.rpartition(":")
        return (host or default_host), int(port)
    return default_host, int(s)


def ssh_tunnel_args(remote: str, remote_port: int, local_port: int, bind: str = "127.0.0.1",
                    ssh_options: list[str] | None = None) -> list[str]:
    """The reverse-tunnel command (matches monet's express-proxy): exposes
    ``remote_port`` on the node's ``bind`` address → laptop's relay ``local_port``.

    ``bind`` defaults to the node's loopback (``127.0.0.1``) — right when the agent
    runs *on* the node or with host networking. For agents in a container that can't
    see the node's loopback, use ``0.0.0.0`` so other node interfaces (the docker
    bridge gateway) can reach it — this needs ``GatewayPorts yes`` (or
    ``clientspecified``) in the node's sshd config.
    """
    return [
        "ssh", "-N",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        *(ssh_options or []),  # e.g. -J jumphost, -i key, -p 2222 — `remote` may be user@ip
        "-R", f"{bind}:{remote_port}:127.0.0.1:{local_port}",
        remote,
    ]


# --- relay (laptop) ----------------------------------------------------------


def _upstream_ssl_context(*, cafile: str | None = None, insecure: bool = False) -> ssl.SSLContext:
    """TLS context for the relay→gateway hop. Verifies by default (like monet's Node
    client). Node ships its own CA bundle; a laptop's Python may not have the gateway's
    issuer (macOS especially) → "unable to get local issuer certificate". Fix it right
    with a real bundle (`cafile`), or bypass verification (`insecure`) for a quick
    unblock over the already-authenticated ssh tunnel."""
    if insecure:
        return ssl._create_unverified_context()  # noqa: S323 — opt-in, documented
    return ssl.create_default_context(cafile=cafile)  # cafile=None → system defaults


def _make_handler(*, upstream: str, pool: KeyPool, max_concurrent: int, connect=None,
                  cafile: str | None = None, insecure: bool = False, debug: bool = False):
    host, port, base_path = split_upstream(upstream)
    if connect is None:  # default: real HTTPS to the gateway (tests inject a fake)
        ssl_ctx = _upstream_ssl_context(cafile=cafile, insecure=insecure)

        def connect():
            return http.client.HTTPSConnection(host, port, context=ssl_ctx, timeout=30 * 60)

    active = {"n": 0}
    lock = threading.Lock()

    class Relay(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "gateway-proxy/1.0"

        def log_message(self, *a):  # noqa: ANN002 — quiet; we print our own lines
            pass

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def do_GET(self) -> None:  # noqa: N802 — health probe for `check`
            code = 200 if urlsplit(self.path).path == "/" else 404
            self._json(code, {"status": "ok" if code == 200 else "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path not in ALLOWED_PATHS:
                self._json(404, {"error": f"Unsupported proxy path: {path}"})
                return
            with lock:
                if active["n"] >= max_concurrent:
                    self._json(503, {"error": "gateway proxy overloaded"})
                    return
                active["n"] += 1
            try:
                self._relay(path)
            finally:
                with lock:
                    active["n"] -= 1

        def _relay(self, path: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            auth = self.headers.get("Authorization")
            if not auth:
                key = pool.next()
                auth = f"Bearer {key}" if key else None
            if not auth:
                self._json(401, {"error": "no Authorization header and no fallback key"})
                return
            headers = forward_headers(dict(self.headers), authorization=auth, host=host)
            conn = connect()
            # --- diag: attribute where a mid-stream drop originates ---------------------
            # A read that ends early / raises  = the GATEWAY hung up (upstream).
            # A write that raises              = the CLIENT/tunnel side dropped (downstream).
            # One [relay-diag] line per request so a load test NAMES the culprit. Observability
            # only — the stream/502/close behavior below is unchanged.
            t0 = time.monotonic()
            nbytes = 0
            saw_done = False
            reason = "ok"
            try:
                conn.request("POST", base_path + self.path, body=body, headers=headers)
                upstream_res = conn.getresponse()
                # Stream the body back; Connection: close so the client reads to EOF
                # (handles SSE without re-chunking).
                self.send_response(upstream_res.status)
                for k, v in upstream_res.getheaders():
                    if k.lower() not in _HOP_BY_HOP:
                        self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                while True:
                    try:
                        chunk = upstream_res.read(65536)
                    except Exception as e:  # noqa: BLE001 — GATEWAY read failed
                        reason = f"upstream_read_error:{type(e).__name__}"
                        break
                    if not chunk:
                        reason = "upstream_eof" if saw_done else "upstream_eof_EARLY"
                        break
                    nbytes += len(chunk)
                    if b"[DONE]" in chunk:
                        saw_done = True
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except Exception as e:  # noqa: BLE001 — CLIENT/tunnel write failed
                        reason = f"client_write_error:{type(e).__name__}"
                        break
            except Exception as e:  # noqa: BLE001 — surface upstream failure as 502
                reason = f"connect_error:{type(e).__name__}"
                try:
                    self._json(502, {"error": f"upstream: {type(e).__name__}: {e}"})
                except Exception:
                    pass
            finally:
                conn.close()
                if debug:  # per-request termination reason (upstream vs client); off by default
                    try:
                        sys.stderr.write(
                            f"[relay-diag] {reason} dur={time.monotonic() - t0:.1f}s "
                            f"bytes={nbytes} done={saw_done} path={path}\n")
                        sys.stderr.flush()
                    except Exception:  # noqa: BLE001 — logging must never break the relay
                        pass

    return Relay


def start_relay(*, local_port: int, upstream: str, pool: KeyPool, max_concurrent: int,
                connect=None, cafile: str | None = None, insecure: bool = False,
                debug: bool = False) -> tuple[ThreadingHTTPServer, int]:
    """Bind the relay on 127.0.0.1, auto-incrementing the port a few times if taken.
    Returns (server, bound_port)."""
    handler = _make_handler(upstream=upstream, pool=pool, max_concurrent=max_concurrent,
                            connect=connect, cafile=cafile, insecure=insecure, debug=debug)
    last: OSError | None = None
    for candidate in range(local_port, local_port + 50):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), handler)
        except OSError as e:  # port in use
            last = e
            continue
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, server.server_address[1]  # actual bound port (0 → OS-assigned)
    raise SystemExit(f"could not bind a relay port near {local_port}: {last}")


# --- tunnel (laptop) ---------------------------------------------------------


def run_tunnel(*, remote: str, remote_port: int, local_port: int, reconnect: bool,
               max_attempts: int, bind: str = "127.0.0.1", ssh_options: list[str] | None = None) -> int:
    """Open `ssh -R`, blocking; on drop, reconnect (same ports → same URL) with
    exponential backoff. The attempt counter resets after a 60s stable window."""
    args = ssh_tunnel_args(remote, remote_port, local_port, bind=bind, ssh_options=ssh_options)
    url = f"http://127.0.0.1:{remote_port}/"
    delay, attempts = 1.0, 0
    while True:
        print(f"[gateway-proxy] tunnel up: {url} (ssh -R → {remote})", flush=True)
        started = time.monotonic()
        try:
            subprocess.run(args)  # blocks until ssh exits
        except KeyboardInterrupt:
            return 0
        if not reconnect:
            print("[gateway-proxy] tunnel closed (--no-reconnect).", flush=True)
            return 0
        if time.monotonic() - started > 60:  # was stable → reset backoff
            delay, attempts = 1.0, 0
        attempts += 1
        if max_attempts and attempts >= max_attempts:
            print(f"[gateway-proxy] giving up after {attempts} reconnect attempts.", flush=True)
            return 1
        print(f"[gateway-proxy] tunnel dropped; reconnecting in {delay:.0f}s (preserving {url})", flush=True)
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            return 0
        delay = min(delay * 2, 30.0)


# --- routable forwarder (node) — the agent-agnostic express_forward.sh -------


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _handle_forward(client: socket.socket, to_host: str, to_port: int) -> None:
    try:
        upstream = socket.create_connection((to_host, to_port), timeout=10)
    except OSError:
        client.close()
        return
    # create_connection LEAVES the 10s timeout on the socket; without this, _pipe's recv()
    # raises socket.timeout after 10s of no data and kills the stream — fatal for a reasoning
    # model that idles >10s before its first token. The 10s is a CONNECT budget only; make the
    # established connection blocking so idle gaps don't close it.
    upstream.settimeout(None)
    threading.Thread(target=_pipe, args=(upstream, client), daemon=True).start()
    _pipe(client, upstream)  # this thread pumps client→upstream
    client.close()
    upstream.close()


def run_forwarder(*, listen: str, to: str) -> int:
    """TCP forwarder exposing a loopback-bound tunnel on a routable interface (the
    node's docker-bridge / IP) so containers reach it — the agent-agnostic
    equivalent of ``express_forward.sh``'s socat (no socat, no sshd GatewayPorts).

    ``to`` may be a COMMA-SEPARATED list of ``host:port`` upstreams; each accepted
    connection is round-robined across them so N parallel ssh tunnels share the load
    (no single ssh connection carries every stream). A single target = old behavior."""
    lh, lp = hostport(listen)
    targets = [hostport(t, default_host="127.0.0.1") for t in to.split(",") if t.strip()]
    rr, rr_lock = itertools.cycle(targets), threading.Lock()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((lh, lp))
    srv.listen(128)
    print(f"[gateway-proxy] forwarding {lh}:{lp} → round-robin {targets} (Ctrl-C to stop)", flush=True)
    try:
        while True:
            client, _ = srv.accept()
            with rr_lock:
                th, tp = next(rr)
            threading.Thread(target=_handle_forward, args=(client, th, tp), daemon=True).start()
    except KeyboardInterrupt:
        return 0
    finally:
        srv.close()


# --- subcommands -------------------------------------------------------------


def cmd_forward(args: argparse.Namespace) -> int:
    return run_forwarder(listen=args.listen, to=args.to)


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    pool = KeyPool(parse_keys(os.environ.get(args.key_env) or os.environ.get("LLM_GATEWAY_EXPRESS_API_KEY")))
    if not pool._keys:
        print(f"[gateway-proxy] note: no keys in ${args.key_env} — relay will only pass "
              f"requests that carry their own Authorization header.", flush=True)
    _server, local_port = start_relay(
        local_port=args.local_port, upstream=args.upstream, pool=pool, max_concurrent=args.max_concurrent,
        cafile=args.cafile or None, insecure=args.insecure, debug=args.debug,
    )
    if args.insecure:
        print("[gateway-proxy] WARNING: --insecure — upstream TLS verification is OFF.", flush=True)
    remote_port = args.remote_port or local_port
    print(f"[gateway-proxy] relay → {args.upstream}  (127.0.0.1:{local_port}, {len(pool._keys)} key(s))", flush=True)
    print("\n[gateway-proxy] on the CLUSTER node, set (the agent forwards these):")
    print(f"    export LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL=http://127.0.0.1:{remote_port}/")
    print(f"    export {args.key_env}=\"<same key list as here>\"\n")
    return run_tunnel(
        remote=args.remote, remote_port=remote_port, local_port=local_port,
        reconnect=not args.no_reconnect, max_attempts=args.reconnect_max_attempts,
        bind=args.remote_bind, ssh_options=args.ssh_option,
    )


def cmd_check(args: argparse.Namespace) -> int:
    """On the cluster: confirm the proxy is reachable through the tunnel. Any HTTP
    response (even 404 to `/`) means relay + tunnel are up."""
    import os

    url = args.url or os.environ.get("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL") or f"http://127.0.0.1:{DEFAULT_PORT}/"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code  # a 404 still proves reachability
    except Exception as e:  # noqa: BLE001
        print(f"[gateway-proxy] NOT reachable at {url}: {type(e).__name__}: {e}")
        print("  is `serve` running on your laptop with --remote <this node>?")
        return 1
    print(f"[gateway-proxy] reachable at {url} (HTTP {code}) — tunnel + relay are up.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gateway_proxy.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="laptop: run the relay + reverse tunnel")
    s.add_argument("--remote", required=True,
                   help="ssh target: config alias, host, or user@ip (e.g. ubuntu@10.0.130.17)")
    s.add_argument("--ssh-option", action="append", default=[], metavar="OPT",
                   help="extra ssh args, repeatable (e.g. --ssh-option -J --ssh-option jumphost, or -i key)")
    s.add_argument("--local-port", type=int, default=DEFAULT_PORT, help="relay port on the laptop")
    s.add_argument("--remote-port", type=int, default=0, help="port exposed on the node (default: = local)")
    s.add_argument("--upstream", default=DEFAULT_UPSTREAM, help="the LLM Gateway Express base URL")
    s.add_argument("--cafile", default="", metavar="PATH",
                   help="CA bundle to verify the gateway cert (fixes 'unable to get local issuer certificate')")
    s.add_argument("--insecure", action="store_true",
                   help="skip upstream TLS verification (quick unblock; the ssh tunnel still protects laptop↔node)")
    s.add_argument("--key-env", default="LLM_GATEWAY_EXPRESS_API_KEY_LIST", help="env var holding the key list")
    s.add_argument("--max-concurrent", type=int, default=256)
    s.add_argument("--debug", action="store_true",
                   help="log a per-request [relay-diag] line (upstream vs client termination) to stderr")
    s.add_argument("--remote-bind", default="127.0.0.1",
                   help="node bind addr for the exposed port; use 0.0.0.0 if the agent runs in a "
                        "container that can't see the node's loopback (needs sshd GatewayPorts)")
    s.add_argument("--no-reconnect", action="store_true", help="exit when the tunnel drops")
    s.add_argument("--reconnect-max-attempts", type=int, default=0, help="0 = unlimited")
    s.set_defaults(func=cmd_serve)

    f = sub.add_parser("forward", help="node: expose the loopback tunnel on a routable interface")
    f.add_argument("--listen", default=f"0.0.0.0:{DEFAULT_PORT + 8}",
                   help="host:port to expose (default 0.0.0.0:18088; reachable as <node-ip>:18088)")
    f.add_argument("--to", default=f"127.0.0.1:{DEFAULT_PORT}",
                   help="the reverse-tunnel port on the node's loopback (default 127.0.0.1:18080)")
    f.set_defaults(func=cmd_forward)

    c = sub.add_parser("check", help="cluster: verify the proxy is reachable")
    c.add_argument("--url", default="", help="proxy URL (default: $LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL)")
    c.set_defaults(func=cmd_check)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
