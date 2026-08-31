"""Entrypoint for the in-sandbox stub.

Two transports (the Docker backend picks one per host platform):

- ``--uds /run/xrlenv/stub.sock`` — Linux production default; the host opens
  the bind-mounted socket directly.
- ``--bind-host 0.0.0.0 --bind-port <p>`` — macOS / Windows fallback (Docker
  Desktop's host↔VM bridge does not route uds connections, only file
  presence). The container publishes ``<p>/tcp`` which the host reaches via
  ``127.0.0.1:<ephemeral>``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from xrlenv.observability.logging import configure_logging
from xrlenv.sandbox_stub.server import StubServer

LOGGER = logging.getLogger("xrlenv.sandbox_stub")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="xrlenv-stub")
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument(
        "--uds",
        default=os.environ.get("XRLENV_STUB_UDS"),
        help="Unix-domain-socket path to bind on (Linux production default)",
    )
    transport.add_argument(
        "--bind-port",
        type=int,
        default=None,
        help="TCP port to bind on (use with --bind-host)",
    )
    parser.add_argument(
        "--bind-host",
        default="0.0.0.0",
        help="TCP host to bind on (only used with --bind-port; default 0.0.0.0)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("XRLENV_STUB_LOG_LEVEL", "INFO"),
        help="Python logging level (default INFO)",
    )
    parser.add_argument(
        "--log-format",
        choices=("auto", "json", "pretty"),
        default=os.environ.get("XRLENV_STUB_LOG_FORMAT", "auto"),
        help=(
            "Log output style. The stub's stdout is normally tail-captured "
            "into the run dir, so 'auto' (default) lands JSON envelopes "
            "that parse alongside the node-agent + control-plane records."
        ),
    )
    args = parser.parse_args(argv)
    if not args.uds and not args.bind_port:
        # Fall back to the env-default uds if nothing was explicitly chosen.
        args.uds = "/run/xrlenv/stub.sock"
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    # Stub stdout is normally captured to a file (no TTY), so 'auto'
    # resolves to JSON and ``stub.log`` parses uniformly with the
    # node-agent + control-plane records (spec 08 §"Structured logs").
    configure_logging(level=args.log_level, log_format=args.log_format)

    if args.bind_port is not None:
        LOGGER.info("starting stub on tcp=%s:%d", args.bind_host, args.bind_port)
        server = StubServer(bind_host=args.bind_host, bind_port=args.bind_port)
    else:
        LOGGER.info("starting stub on uds=%s", args.uds)
        server = StubServer(uds_path=args.uds)
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        LOGGER.info("interrupted; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
