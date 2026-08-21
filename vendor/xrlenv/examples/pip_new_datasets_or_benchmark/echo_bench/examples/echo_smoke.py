"""echo_bench oracle smoke driver.

Three rollouts, oracle policy: read ``target`` from the initial
observation, return ``{"output": target}`` as the only action.
Expected: 3 / 3 sealed ``finished`` with ``final_reward = 1.0``.

This driver supports a single mode (``--local``) on purpose — the
worked example is meant to validate B11.6 + D22 end-to-end on a
laptop in ~30 s. For multi-VM smoke patterns see
``xrlenv_plugins/benchmarks/terminal_bench_2/examples/tb2_acceptance_smoke.py``.

Usage::

    .venv/bin/python examples/pip_new_datasets_or_benchmark/echo_bench/examples/echo_smoke.py --local
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from xrlenv import Client
from xrlenv.control import build_local_runtime

LOGGER = logging.getLogger("xrlenv.smoke.echo_bench")

DEFAULT_INSTANCES: tuple[str, ...] = (
    "echo-hello",
    "echo-multiline",
    "echo-symbols",
)


def _make_oracle_policy() -> Any:
    """Return an async policy that echoes the ``target`` field from
    the initial observation. Captures target on first call so the
    second-and-later calls (after done=True propagates) are no-ops.
    """
    state: dict[str, Any] = {"emitted": False, "target": None}

    async def _policy(obs: Any) -> Any:
        if state["emitted"]:
            return {"__exit__": True}
        if not isinstance(obs, dict) or obs.get("kind") != "echo_bench.start":
            raise RuntimeError(
                f"echo_bench oracle: expected echo_bench.start obs, got {obs!r}"
            )
        target = obs.get("target")
        if not isinstance(target, str):
            raise RuntimeError(
                f"echo_bench oracle: missing target string in obs {obs!r}"
            )
        state["emitted"] = True
        state["target"] = target
        return {"output": target}

    return _policy


async def _drive_one(client: Client, instance_id: str) -> tuple[str, Any]:
    LOGGER.info("[%s] starting rollout", instance_id)
    policy = _make_oracle_policy()
    try:
        session = await client.rollout(
            template="echo-bench",
            init={"instance_id": instance_id},
            task_key=instance_id,
        )
    except Exception as exc:
        LOGGER.exception("[%s] rollout start failed", instance_id)
        return instance_id, exc

    try:
        async with session:
            while not session.done:
                action = await policy(session.observation)
                await session.step(action)
        return instance_id, session.trajectory
    except Exception as exc:
        LOGGER.exception("[%s] rollout drive failed", instance_id)
        return instance_id, exc


def _report(results: list[tuple[str, Any]]) -> int:
    """Print a per-rollout summary; return exit code 0 iff all succeeded."""
    successes = 0
    for instance_id, outcome in results:
        if isinstance(outcome, Exception):
            print(
                f"[{instance_id}] FAILED: {type(outcome).__name__}: {outcome}",
                file=sys.stderr,
            )
            continue
        traj = outcome
        reward = getattr(traj, "final_reward", None)
        status = getattr(traj, "status", None)
        node = getattr(traj, "node_id", None)
        ok = reward == 1.0
        if ok:
            successes += 1
        print(
            f"[{instance_id}] status={status} final_reward={reward} "
            f"node={node}{'' if ok else '  ❌'}"
        )
    print(
        f"\n{successes} / {len(results)} rollouts sealed finished+reward=1.0",
        file=sys.stderr,
    )
    return 0 if successes == len(results) else 1


async def _run_local(args: argparse.Namespace) -> int:
    runtime = build_local_runtime()
    await runtime.start()
    try:
        client = Client.in_process(runtime.service)
        results = await asyncio.gather(
            *(_drive_one(client, inst) for inst in args.instances)
        )
    finally:
        await runtime.shutdown()
    return _report(list(results))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="echo_smoke",
        description=(
            "Run the echo_bench oracle smoke against a LocalRuntime. "
            "Expected: 3/3 final_reward=1.0."
        ),
    )
    p.add_argument(
        "--local", action="store_true",
        help="Run against a LocalRuntime in this process (only mode supported).",
    )
    p.add_argument(
        "--instances",
        type=lambda s: tuple(x.strip() for x in s.split(",") if x.strip()),
        default=DEFAULT_INSTANCES,
        help="Comma-separated instance ids (default: all 3).",
    )
    return p


async def _async_main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.local:
        print(
            "echo_smoke: only --local mode is supported in this worked "
            "example. Pass --local.",
            file=sys.stderr,
        )
        return 2
    return await _run_local(args)


def main() -> None:
    raise SystemExit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
