"""Unit tests for the standalone gateway-proxy helpers + a live relay round-trip
against a fake upstream (no SSH, no real gateway). The script is stdlib-only and not
importable as a package, so we load it by path."""

from __future__ import annotations

import importlib.util
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[2] / "scripts" / "gateway" / "gateway_proxy.py"
    spec = importlib.util.spec_from_file_location("gateway_proxy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gp = _load()


# --- pure helpers ------------------------------------------------------------


def test_parse_keys_splits_and_dedupes() -> None:
    assert gp.parse_keys("a, b; c  d") == ["a", "b", "c", "d"]
    assert gp.parse_keys("k1,k1,k2") == ["k1", "k2"]
    assert gp.parse_keys("") == [] and gp.parse_keys(None) == []


def test_key_pool_round_robin() -> None:
    pool = gp.KeyPool(["k1", "k2", "k3"])
    assert [pool.next() for _ in range(4)] == ["k1", "k2", "k3", "k1"]
    assert gp.KeyPool([]).next() is None


def test_split_upstream() -> None:
    assert gp.split_upstream("https://host.example/") == ("host.example", 443, "")
    assert gp.split_upstream("https://h:8443/base/") == ("h", 8443, "/base")
    with pytest.raises(ValueError, match="https"):
        gp.split_upstream("http://h/")  # must be https


def test_forward_headers_replaces_auth_and_host_no_dupes() -> None:
    incoming = {"Content-Type": "application/json", "Host": "x", "Content-Length": "5",
                "Connection": "keep-alive", "authorization": "old"}
    out = gp.forward_headers(incoming, authorization="Bearer k", host="up.host")
    assert out == {"Content-Type": "application/json", "Authorization": "Bearer k", "Host": "up.host"}


def test_ssh_tunnel_args_matches_reverse_tunnel_contract() -> None:
    args = gp.ssh_tunnel_args("ubuntu@10.0.130.17", 18080, 19090,
                              ssh_options=["-J", "jump", "-i", "key"])
    assert args[0] == "ssh" and args[-1] == "ubuntu@10.0.130.17" and "-N" in args
    assert "-R" in args and "127.0.0.1:18080:127.0.0.1:19090" in args
    assert "ExitOnForwardFailure=yes" in args and "ServerAliveInterval=30" in args
    # extra ssh options land before the -R refspec; user@ip is the target verbatim.
    assert args.index("-J") < args.index("-R") and "jump" in args and "key" in args


def test_upstream_ssl_context_verify_and_insecure() -> None:
    import ssl

    default = gp._upstream_ssl_context()
    assert default.verify_mode == ssl.CERT_REQUIRED and default.check_hostname is True
    insecure = gp._upstream_ssl_context(insecure=True)
    assert insecure.verify_mode == ssl.CERT_NONE and insecure.check_hostname is False


def test_hostport() -> None:
    assert gp.hostport("1.2.3.4:18088") == ("1.2.3.4", 18088)
    assert gp.hostport(":18088") == ("0.0.0.0", 18088)
    assert gp.hostport("18080", default_host="127.0.0.1") == ("127.0.0.1", 18080)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_forwarder_pipes_bytes() -> None:
    # A routable forwarder (the express_forward.sh equivalent) → an echo target.
    import socket as _socket
    import time

    target = _socket.socket()
    target.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    target.bind(("127.0.0.1", 0))
    target.listen()
    tport = target.getsockname()[1]

    def _echo():
        c, _ = target.accept()
        c.sendall(b"echo:" + c.recv(1024))
        c.close()

    threading.Thread(target=_echo, daemon=True).start()
    fport = _free_port()
    threading.Thread(target=gp.run_forwarder,
                     kwargs={"listen": f"127.0.0.1:{fport}", "to": f"127.0.0.1:{tport}"},
                     daemon=True).start()

    conn = None
    for _ in range(100):  # wait for the forwarder to bind
        try:
            conn = _socket.create_connection(("127.0.0.1", fport), timeout=1)
            break
        except OSError:
            time.sleep(0.02)
    assert conn is not None
    conn.sendall(b"hi")
    out = conn.recv(1024)
    conn.close()
    assert out == b"echo:hi"


# --- live relay round-trip (fake upstream via monkeypatched HTTPSConnection) --


class _FakeUpstream(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: ANN002
        pass

    def do_POST(self):  # noqa: N802 — echo the Authorization back, streamed
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        auth = self.headers.get("Authorization", "")
        payload = f"auth={auth};path={self.path};body={body.decode()}".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def test_relay_injects_key_and_forwards() -> None:
    # Fake "gateway" upstream on plain HTTP; inject it as the relay's connection
    # factory (no global monkeypatching of http.client, which would break urllib).
    import http.client

    up = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    up_host, up_port = up.server_address[0], up.server_address[1]

    server, port = gp.start_relay(
        local_port=0, upstream="https://fake-gateway/", pool=gp.KeyPool(["KEY1", "KEY2"]),
        max_concurrent=8, connect=lambda: http.client.HTTPConnection(up_host, up_port, timeout=5),
    )
    try:
        # No Authorization header → relay injects a pooled key (round-robin).
        req = urllib.request.Request(f"http://127.0.0.1:{port}/chat/completions",
                                     data=b'{"x":1}', method="POST")
        out1 = urllib.request.urlopen(req, timeout=5).read().decode()
        out2 = urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/chat/completions", data=b'{}', method="POST"), timeout=5).read().decode()
        assert "auth=Bearer KEY1" in out1 and "path=/chat/completions" in out1 and 'body={"x":1}' in out1
        assert "auth=Bearer KEY2" in out2  # round-robin to the next key
        # Unsupported path → 404 from the relay itself.
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions", data=b"{}", method="POST"), timeout=5)
        assert e.value.code == 404
    finally:
        server.shutdown()
        up.shutdown()


# --- regression: _json must produce valid JSON for all string values ---------


def test_relay_error_response_is_valid_json() -> None:
    """The _json helper must use json.dumps, not str().replace() — regression for
    the P0 bug where strings containing apostrophes produced malformed JSON."""
    import http.client
    import json as _json

    # Start a relay with no upstream (requests will 404 on unsupported path).
    up = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    up_host, up_port = up.server_address[0], up.server_address[1]

    server, port = gp.start_relay(
        local_port=0, upstream="https://fake-gateway/",
        pool=gp.KeyPool([]),  # empty pool — no keys
        max_concurrent=8, connect=lambda: http.client.HTTPConnection(up_host, up_port, timeout=5),
    )
    try:
        # Empty key pool + no Authorization header → 401 with a plain-text error message.
        # The response body must be valid JSON, not str(dict).replace("'", '"').
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat/completions",
            data=b"{}", method="POST",
        )
        # Expect 401 (HTTPError) with a JSON body we can parse.
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        body = exc_info.value.read()
        parsed = _json.loads(body)  # must not raise — regression guard
        assert "error" in parsed

        # Unsupported path 404 must also be valid JSON.
        req2 = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=b"{}", method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info2:
            urllib.request.urlopen(req2, timeout=5)
        parsed2 = _json.loads(exc_info2.value.read())
        assert "error" in parsed2
    finally:
        server.shutdown()
        up.shutdown()


# --- hostport edge cases -------------------------------------------------------


def test_hostport_port_only_bare_string() -> None:
    assert gp.hostport("8080") == ("0.0.0.0", 8080)
    assert gp.hostport("8080", default_host="127.0.0.1") == ("127.0.0.1", 8080)


def test_hostport_colon_only_port() -> None:
    assert gp.hostport(":18080") == ("0.0.0.0", 18080)


def test_hostport_port_zero_os_assigned() -> None:
    assert gp.hostport("0") == ("0.0.0.0", 0)
    assert gp.hostport("127.0.0.1:0") == ("127.0.0.1", 0)


# --- ssh_tunnel_args: bind override -------------------------------------------


def test_ssh_tunnel_args_bind_0000_for_container_reach() -> None:
    """bind=0.0.0.0 produces 0.0.0.0:<remote>:127.0.0.1:<local> in the -R refspec."""
    args = gp.ssh_tunnel_args("mynode", 18080, 19090, bind="0.0.0.0")
    r_idx = args.index("-R")
    assert args[r_idx + 1] == "0.0.0.0:18080:127.0.0.1:19090"


def test_ssh_tunnel_args_no_options() -> None:
    """ssh_options=None omits extra args (no empty list spliced in)."""
    args = gp.ssh_tunnel_args("node", 18080, 19090)
    # Only the fixed options should be there (ExitOnForwardFailure, ServerAlive×2, -R)
    assert "-J" not in args
    assert "-i" not in args
