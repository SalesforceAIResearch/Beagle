"""In-sandbox stub server (spec 01).

A small Python server that runs inside every sandbox as PID 1's child (under
``tini``). Exposes an HTTP/1.1 surface on a unix domain socket; the wire format
is a strict subset of E2B's REST API so any tool that targets E2B works with no
extra glue.
"""

from xrlenv.sandbox_stub.server import StubServer, build_app

__all__ = ["StubServer", "build_app"]
