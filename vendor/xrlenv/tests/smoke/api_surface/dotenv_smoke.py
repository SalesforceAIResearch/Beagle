"""End-to-end smoke for ``xrlenv.client.dotenv`` helpers.

Proves both secret-injection shapes work against a **real** Docker
container, not just the fake-session unit tests:

1. ``parse_dotenv(path)`` →
   ``Client.acquire_container(environment=...)`` →
   ``echo $KEY`` inside the container returns the value the
   operator's ``.env`` declared.

2. ``upload_dotenv(session, source=...)`` →
   ``cat /workspace/.env`` inside the container returns the file
   the operator handed to the helper, byte-for-byte.

This is the live-cluster counterpart to
``tests/unit/client/test_dotenv.py``. The unit tests verify call
shape (mkdir-then-put-archive, tarball contents, parser shapes);
this smoke verifies that the Docker side actually honors the
``environment=`` dict at create time and that ``put_archive``
extraction lands the file at the documented path.

Operator setup
==============

Same as the sibling ``raw_container_smoke.py`` — the script dials
an existing cluster via ``--connect-host`` / ``--connect-port`` or
runs in-process via ``--in-process``. ``$XRLENV_CONSUMER_TOKEN`` is
honored when set.

Usage::

    # In-process (no real cluster):
    .venv/bin/python tests/smoke/api_surface/dotenv_smoke.py --in-process

    # Against an existing cluster:
    .venv/bin/python tests/smoke/api_surface/dotenv_smoke.py \\
        --connect-host 127.0.0.1 --connect-port 50051

The smoke writes a tiny ``.env`` to a temp file, runs both helpers
against an ``alpine:3.19`` container, asserts the in-container
readback matches, then destroys the container. Exit 0 = pass.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
from pathlib import Path

from xrlenv import Client
from xrlenv.client.dotenv import parse_dotenv, upload_dotenv

LOGGER = logging.getLogger("xrlenv.smoke.dotenv")


# ──────────────────────────────────────────────────────────────────────────────
# Test payload — small enough to inline, exercises the parser's tricky shapes.
# ──────────────────────────────────────────────────────────────────────────────


_DOTENV_BODY = (
    "# operator's API keys + odd shapes\n"
    "\n"
    "ANTHROPIC_API_KEY=sk-ant-test-value-123\n"
    "QUOTED_VAR=\"hello world\"\n"
    "export EXPORTED_VAR=exported-value\n"
    "EMPTY_VAR=\n"
    "DATABASE-URL=skipped-because-of-hyphen\n"
)

# Keys the smoke expects parse_dotenv to keep + their expected values.
_EXPECTED_PARSED = {
    "ANTHROPIC_API_KEY": "sk-ant-test-value-123",
    "QUOTED_VAR": "hello world",
    "EXPORTED_VAR": "exported-value",
    "EMPTY_VAR": "",
}


# ──────────────────────────────────────────────────────────────────────────────
# Smoke body.
# ──────────────────────────────────────────────────────────────────────────────


async def _run_against_client(
    client: Client, *, image: str,
) -> int:
    """Acquire one alpine container, exercise both helpers, assert
    the in-container readback. Returns exit code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env_path = Path(tmpdir) / ".env"
        env_path.write_text(_DOTENV_BODY, encoding="utf-8")

        # ── Step 1: parse_dotenv unit sanity ────────────────────────────
        parsed = parse_dotenv(env_path)
        if parsed != _EXPECTED_PARSED:
            LOGGER.error(
                "parse_dotenv produced %r; expected %r",
                parsed, _EXPECTED_PARSED,
            )
            return 1
        LOGGER.info("parse_dotenv: %d keys parsed cleanly", len(parsed))

        # ── Step 2: env vars land in the container at create time ──────
        LOGGER.info(
            "acquiring %s with environment=%d keys", image, len(parsed),
        )
        async with await client.acquire_container(
            image=image,
            command=["sleep", "infinity"],
            environment=parsed,
            labels={"workflow": "dotenv-smoke"},
        ) as session:
            LOGGER.info(
                "container ready rollout=%s node=%s",
                session.rollout_id, session.node_id,
            )

            for key, expected in parsed.items():
                # ``printf %s "$VAR"`` avoids the trailing newline ``echo``
                # adds, which simplifies the byte-level compare.
                result = await session.exec(
                    ["sh", "-c", f'printf %s "${key}"'],
                    timeout_s=15,
                )
                actual = result.stdout.decode("utf-8")
                if result.exit_code != 0:
                    LOGGER.error(
                        "exec for %s exited %d; stderr=%r",
                        key, result.exit_code, result.stderr,
                    )
                    return 1
                if actual != expected:
                    LOGGER.error(
                        "in-container %s = %r; expected %r",
                        key, actual, expected,
                    )
                    return 1
            LOGGER.info(
                "env-vars-at-acquire: all %d keys readback matched",
                len(parsed),
            )

            # ── Step 3: upload_dotenv landed a file at the documented path ──
            landed_at = await upload_dotenv(
                session, source=env_path,
            )
            if landed_at != "/workspace/.env":
                LOGGER.error("upload_dotenv returned %r; expected /workspace/.env", landed_at)
                return 1
            read_result = await session.exec(
                ["cat", "/workspace/.env"],
                timeout_s=15,
            )
            if read_result.exit_code != 0:
                LOGGER.error(
                    "cat /workspace/.env exited %d; stderr=%r",
                    read_result.exit_code, read_result.stderr,
                )
                return 1
            in_container = read_result.stdout.decode("utf-8")
            if in_container != _DOTENV_BODY:
                LOGGER.error(
                    "file readback mismatch.\nWROTE:\n%s\nGOT:\n%s",
                    _DOTENV_BODY, in_container,
                )
                return 1
            LOGGER.info(
                "upload_dotenv: file landed at %s, bytes match", landed_at,
            )
    print("\n[dotenv-smoke] PASS — both shapes work end-to-end")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI / topology.
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dotenv_smoke",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--connect-host", default=None,
        help="Existing cluster host. Mutually exclusive with --in-process.",
    )
    p.add_argument(
        "--connect-port", type=int, default=50051,
        help="gRPC port (default 50051).",
    )
    p.add_argument(
        "--consumer-token", default=None,
        help="Consumer bearer (or $XRLENV_CONSUMER_TOKEN).",
    )
    p.add_argument(
        "--in-process", action="store_true", default=False,
        help="Run against an in-process LocalRuntime instead of a real cluster.",
    )
    p.add_argument(
        "--image", default="alpine:3.19",
        help="Container image to acquire (default alpine:3.19; small + has sh).",
    )
    return p


async def _run(args: argparse.Namespace) -> int:
    if args.in_process:
        from xrlenv.control.runtime import build_local_runtime
        runtime = build_local_runtime(
            node_id="dotenv-smoke-local",
            run_dir_retention_days=None,
            metrics_port=None,
        )
        await runtime.start()
        try:
            client = Client.in_process(runtime.service)
            return await _run_against_client(client, image=args.image)
        finally:
            await runtime.shutdown()
    if not args.connect_host:
        raise SystemExit(
            "pass --in-process for a self-contained run, or "
            "--connect-host <host> [--connect-port <port>] to dial an "
            "existing cluster.",
        )
    import os
    token = args.consumer_token or os.environ.get("XRLENV_CONSUMER_TOKEN")
    client = Client.grpc(
        host=args.connect_host, port=args.connect_port, token=token,
    )
    try:
        return await _run_against_client(client, image=args.image)
    finally:
        await client.close()


def main() -> int:
    args = _build_parser().parse_args()
    from xrlenv.observability.logging import configure_logging
    configure_logging()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
