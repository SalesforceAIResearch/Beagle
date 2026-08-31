"""Prometheus ``/metrics`` HTTP exposer (spec 08).

prometheus-client ships ``start_http_server``, but it serves *only* the raw
text exposition format. We want the same endpoint to stay a valid Prometheus
scrape target while also rendering a human-readable dashboard when a browser
opens it. So instead of the stock exposer we build a small dispatching WSGI
app (served on prometheus-client's own ``ThreadingWSGIServer`` so behaviour —
threading, IPv4/IPv6 family selection, silenced request logging — matches the
stock server) and wrap it in a class with ``.start()`` / ``.stop()`` for the
runtime's lifecycle hooks.

Routing on ``/metrics``:

* Prometheus / curl (``Accept: text/plain``, ``*/*``, OpenMetrics) → the raw
  text exposition, delegated unchanged to prometheus-client's own WSGI app so
  OpenMetrics negotiation and gzip still work.
* A browser (``Accept: text/html``) → the rendered dashboard
  (:func:`xrlenv.observability.dashboard.render_dashboard_html`).
* ``?format=raw`` / ``?format=html`` force either side regardless of headers.

``/`` 302-redirects to ``/metrics``; anything else is 404.

Phase 0 binds to ``127.0.0.1:9090`` by default — operators scrape locally via
``curl 127.0.0.1:9090/metrics`` or just open it in a browser. Spec 19 phase-0
leaves this endpoint without auth (loopback-only); a phase-1 admin-bind guard
moves it behind the same bearer-token surface as the gRPC endpoint.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

# These helpers are prometheus-client internals, but they're exactly what its
# own ``start_wsgi_server`` uses; reusing them keeps our exposer's networking
# behaviour identical to the stock one rather than reimplementing it.
from prometheus_client.exposition import (
    ThreadingWSGIServer,
    _get_best_family,
    _SilentHandler,
    make_wsgi_app,
)

from xrlenv.observability.dashboard import prefers_html, render_dashboard_html
from xrlenv.observability.metrics import MetricsRegistry, get_default_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from wsgiref.types import StartResponse, WSGIEnvironment

LOGGER = logging.getLogger(__name__)

_DEFAULT_REFRESH_S = 5
_MAX_REFRESH_S = 3600


def _clamp_refresh(raw: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_REFRESH_S
    return max(0, min(value, _MAX_REFRESH_S))


def build_wsgi_app(
    registry: MetricsRegistry,
    *,
    admin_port: int | None = None,
) -> Callable[[WSGIEnvironment, StartResponse], Iterable[bytes]]:
    """Build the dispatching WSGI app for one :class:`MetricsRegistry`.

    ``admin_port`` (when the admin panel is running) is forwarded to the
    dashboard so its role-clarifier banner can link there for drill-down.
    """
    raw_app = make_wsgi_app(registry.collector_registry)

    def app(
        environ: WSGIEnvironment, start_response: StartResponse
    ) -> Iterable[bytes]:
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/").rstrip("/") or "/"

        if method not in ("GET", "HEAD"):
            start_response(
                "405 Method Not Allowed",
                [
                    ("Content-Type", "text/plain; charset=utf-8"),
                    ("Allow", "GET, HEAD"),
                ],
            )
            return [b"405 Method Not Allowed\n"]

        if path == "/":
            start_response(
                "302 Found",
                [
                    ("Location", "/metrics"),
                    ("Content-Type", "text/plain; charset=utf-8"),
                ],
            )
            return [b"See /metrics\n"]

        if path != "/metrics":
            start_response(
                "404 Not Found", [("Content-Type", "text/plain; charset=utf-8")]
            )
            return [b"404 Not Found - try /metrics\n"]

        params = parse_qs(environ.get("QUERY_STRING", ""))
        fmt = params.get("format", [""])[0].strip().lower()
        accept = environ.get("HTTP_ACCEPT", "")
        want_html = fmt == "html" or (
            fmt != "raw" and prefers_html(accept)
        )

        if not want_html:
            # Hand the scrape straight to prometheus-client (OpenMetrics + gzip).
            raw_result: Iterable[bytes] = raw_app(environ, start_response)
            return raw_result

        refresh_s = _clamp_refresh(params.get("refresh", [str(_DEFAULT_REFRESH_S)])[0])
        body = render_dashboard_html(
            registry.collector_registry, refresh_s=refresh_s, admin_port=admin_port
        ).encode("utf-8")
        start_response(
            "200 OK",
            [
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(body))),
            ],
        )
        return [b""] if method == "HEAD" else [body]

    return app


class MetricsServer:
    """Lifecycle wrapper around the dispatching ``/metrics`` WSGI app."""

    def __init__(
        self,
        registry: MetricsRegistry | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 9090,
        admin_port: int | None = None,
    ) -> None:
        self._registry = registry or get_default_registry()
        self._host = host
        self._port = port
        self._admin_port = admin_port
        self._server: ThreadingWSGIServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        """Bind ``host:port`` and start serving ``/metrics``.

        Safe to call multiple times — a second start with the server already
        running is a no-op. When ``port=0`` was requested, the kernel picks a
        free port and :py:attr:`port` is updated to the actual bound value
        (useful in tests).
        """
        if self._server is not None:
            return

        app = build_wsgi_app(self._registry, admin_port=self._admin_port)

        # Per-bind subclass so we can set the address family (IPv4/IPv6) for
        # this host without mutating the shared class — mirrors prometheus
        # -client's own ``start_wsgi_server``.
        class _Server(ThreadingWSGIServer):
            pass

        _Server.address_family, addr = _get_best_family(  # type: ignore[no-untyped-call]
            self._host, self._port
        )
        httpd = make_server(
            addr, self._port, app, _Server, handler_class=_SilentHandler
        )
        self._server = httpd
        # When the kernel auto-assigns a port (port=0), surface the real value
        # so callers / tests can scrape it.
        self._port = httpd.server_address[1]

        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self._thread = thread

        LOGGER.info(
            "metrics server listening on http://%s:%d/metrics "
            "(HTML dashboard for browsers, raw text for Prometheus)",
            self._host,
            self._port,
        )

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None


def start_metrics_server(
    *,
    registry: MetricsRegistry | None = None,
    host: str = "127.0.0.1",
    port: int = 9090,
    admin_port: int | None = None,
) -> MetricsServer:
    """Convenience: build and start a :class:`MetricsServer` in one call."""
    server = MetricsServer(
        registry=registry, host=host, port=port, admin_port=admin_port
    )
    server.start()
    return server


__all__ = ["MetricsServer", "build_wsgi_app", "start_metrics_server"]
