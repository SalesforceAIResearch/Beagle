"""Slice-1 smoke example: spin a local runtime, run one rollout end-to-end.

Prerequisites:
- Docker daemon reachable from the local user
- ``xrlenv/hello-shell:0.1`` image built (``docker build -t xrlenv/hello-shell:0.1
  xrlenv/templates/hello_shell``)

Run::

    uv run python tests/smoke/single_rollout.py

Expected output: a sequence of step results from echoing strings inside the
sandbox, then a sealed Trajectory with status=finished.
"""

from __future__ import annotations

import asyncio
import logging
import pprint
from pathlib import Path

from xrlenv import Client
from xrlenv.control import build_local_runtime
from xrlenv.observability.logging import configure_json_logging

# Point the client at the hello-shell plug-in's example run-config.
# In real use this would be a file you've copied + edited under
# ~/my-experiments/. See xrlenv/templates/hello_shell/examples/default.run-config.yaml
# for the layout.
_HELLO_SHELL_RUN_CONFIG = (
    Path(__file__).resolve().parent.parent
    / "xrlenv" / "templates" / "hello_shell" / "examples" / "default.run-config.yaml"
)


async def main() -> None:
    configure_json_logging(level=logging.INFO)
    runtime = build_local_runtime()
    await runtime.start()
    client = Client.in_process(runtime.service, run_config=_HELLO_SHELL_RUN_CONFIG)

    try:
        session = await client.rollout(
            template="hello-shell",
            init={"max_steps": 3},
        )
        async with session:
            action = {"cmd": "pwd"}
            result = await session.step(action)
            print(f"Initial obs={result.obs}")
            while not session.done:
                step_index = session.steps_taken
                action = {"cmd": f"echo 'step {step_index}'"}
                result = await session.step(action)
                print(f"step={step_index} reward={result.reward} obs={result.obs}")

        print("\n=== Trajectory ===")
        pprint.pp(session.trajectory)
    finally:
        await client.close()
        await runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
