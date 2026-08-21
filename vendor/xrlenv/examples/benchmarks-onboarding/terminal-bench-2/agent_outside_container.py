"""Pattern 1: outside-the-container agent on a real TB-2 task.

The agent's policy lives **in this process** — typically the same
host as a trainer in an RL setup. It picks an action, hands the
action to xrlenv, gets an observation back, picks the next action.
xrlenv only contributes ``Client.acquire_container(...)`` +
``session.exec(...)`` + ``session.put_archive(...)`` — the same
primitives any consumer would use.

This example walks an **oracle agent** through one terminal-bench-2
task (default ``fix-git``). The oracle reads the task's
``solution/solve.sh`` from the local harbor cache and emits it as a
structured action; xrlenv runs it inside the task's container; we
then push the task's ``tests/`` directory in and run the verifier
exactly the way harbor's runner would, asserting the same
``/logs/verifier/reward.txt == 1`` contract that harbor's report
aggregator consumes downstream.

What this example demonstrates
==============================

- **Outside-agent loop shape.** The agent class owns the loop; it
  decides what to do next and what to do with each observation.
  xrlenv is just the substrate it pokes commands at. Swap
  ``OracleAgent`` for any other policy with the same shape.
- **Per-task image.** Unlike a hello-world demo, the example loads
  the **real** task image declared in the task's ``task.toml``
  (e.g. ``alexgshaw/fix-git:20251031``). The image must be present
  on the chosen node — pre-pull via the operator's image-distribution
  flow (cluster mode) or rely on Docker's on-demand pull (local
  mode).
- **Verifier contract honored verbatim.** The example pushes the
  task's ``tests/`` directory into the container, mkdirs
  ``/logs/verifier``, runs ``bash /tests/test.sh``, and reads
  ``/logs/verifier/reward.txt``. That's byte-compatible with what
  harbor's report aggregator expects, so the pass/fail signal
  matches harbor's own grading.

Prerequisites
=============

The task's solution + verifier need to be in the local harbor task
cache. If they aren't, populate via::

    bash examples/benchmarks-onboarding/terminal-bench-2/scripts/populate-harbor-cache.sh

The script exits cleanly if the cache is already populated.

Cluster-mode operator setup is the same as ``smoke.py``: a control
plane reachable at ``XRLENV_GRPC_HOST:XRLENV_GRPC_PORT`` with at
least one node attached, and the task's image
(``hb__<task_id>`` or the upstream ``alexgshaw/<task>:<rev>`` mirror)
available on that node.

Run with::

    .venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/agent_outside_container.py
    # or pick a different task:
    .venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/agent_inside_container.py --task build-pov-ray
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import io
import json
import os
import sys
import tarfile
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

# Reuse smoke.py's harbor-cache locator so both onboarding paths
# consume the same task layout.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xrlenv import Client
from xrlenv.client.dotenv import parse_dotenv, upload_dotenv

from smoke import SMOKE_TASKS, _locate_task_dir

# ──────────────────────────────────────────────────────────────────────────────
# Task loading.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskBundle:
    """The bits of a TB-2 task this example consumes.

    Mirrors the slice of harbor's task spec that an outside-agent
    needs: the image to run, the natural-language instruction, the
    oracle's solution script, and the verifier files. We deliberately
    don't depend on harbor's TaskConfig type — the goal here is to
    show "you can drive this yourself from raw task files," and the
    file layout is harbor's stable public contract.
    """

    task_id: str
    task_dir: Path
    docker_image: str
    instruction: str
    solve_script: str
    tests_dir: Path


def _load_task_bundle(task_id: str) -> TaskBundle:
    task_dir = _locate_task_dir(task_id)
    toml_text = (task_dir / "task.toml").read_text(encoding="utf-8")
    toml = tomllib.loads(toml_text)
    image = toml.get("environment", {}).get("docker_image")
    if not image:
        raise SystemExit(
            f"task {task_id!r} has no [environment].docker_image in its "
            f"task.toml — the outside-agent example needs an image to "
            f"acquire a container from.",
        )
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    solve_path = task_dir / "solution" / "solve.sh"
    if not solve_path.exists():
        raise SystemExit(
            f"task {task_id!r} has no solution/solve.sh; the oracle "
            f"needs that script as its 'memorized solution.' Pick a "
            f"task that ships an oracle solution, or replace the "
            f"OracleAgent with a real policy.",
        )
    solve_script = solve_path.read_text(encoding="utf-8")
    tests_dir = task_dir / "tests"
    if not tests_dir.is_dir():
        raise SystemExit(
            f"task {task_id!r} has no tests/ directory; cannot run "
            f"the verifier.",
        )
    return TaskBundle(
        task_id=task_id,
        task_dir=task_dir,
        docker_image=image,
        instruction=instruction.strip(),
        solve_script=solve_script,
        tests_dir=tests_dir,
    )


# ──────────────────────────────────────────────────────────────────────────────
# The oracle agent.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Observation:
    """What the agent sees after each action."""

    step: int
    last_action: str | None
    exit_code: int
    stdout: str
    stderr: str


@dataclass
class Action:
    """A bash command line the agent wants xrlenv to execute."""

    bash: str


class OracleAgent:
    """A policy that "knows" the right shell command sequence for a
    TB-2 task. Reads the task's ``solution/solve.sh`` and emits the
    whole script as a single action.

    Real policies would consult a neural network here, look up a
    plan from an LLM call, or run their own per-step reasoning. The
    interface is just ``next_action(observation) -> Action | None``.
    """

    def __init__(self, solve_script: str) -> None:
        self._solve_script = solve_script
        self._fired = False

    def next_action(self, obs: Observation) -> Action | None:
        if self._fired:
            return None
        self._fired = True
        return Action(bash=self._solve_script)


# ──────────────────────────────────────────────────────────────────────────────
# Verifier helpers.
# ──────────────────────────────────────────────────────────────────────────────


def _tar_dir_contents(source_dir: Path) -> bytes:
    """Tar the *contents* of ``source_dir`` (not the directory
    itself) so ``put_archive(target_dir="/tests", tarball=...)``
    lands the files directly under ``/tests/``. Matches the layout
    harbor's verifier expects."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for entry in sorted(source_dir.iterdir()):
            tf.add(entry, arcname=entry.name)
    return buf.getvalue()


def _untar_to_dir(tarball: bytes, target_dir: Path) -> None:
    """Extract a tar bytestream produced by Docker's get_archive
    into ``target_dir``. Mirrors the inverse of
    :func:`_tar_dir_contents` — used to pull the in-container
    verifier outputs (``/logs/verifier``) back to the operator's
    disk for inspection."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r") as tf:
        tf.extractall(path=target_dir, filter="data")


# ──────────────────────────────────────────────────────────────────────────────
# Artifact-dir layout.
# ──────────────────────────────────────────────────────────────────────────────


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_jobs_dir() -> Path:
    return _repo_root() / "tmp"


def _default_run_id(task_id: str) -> str:
    return _dt.datetime.now(_dt.UTC).strftime(
        f"outside-agent-{task_id}-%Y%m%d-%H%M%S",
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI + driver.
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent_outside_container",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task", default=SMOKE_TASKS[0],
        help=(
            f"terminal-bench-2 task id to run (default: "
            f"{SMOKE_TASKS[0]!r}). Must be present in the harbor cache."
        ),
    )
    p.add_argument(
        "--host", default=None,
        help="Control-plane gRPC host. Default: $XRLENV_GRPC_HOST or 127.0.0.1.",
    )
    p.add_argument(
        "--port", type=int, default=None,
        help="Control-plane gRPC port. Default: $XRLENV_GRPC_PORT or 50051.",
    )
    p.add_argument(
        "--verifier-timeout-s", type=float, default=900.0,
        help="Per-task cap on the verifier exec call (default 900s).",
    )
    p.add_argument(
        "--save-artifacts", default=None,
        help=(
            "Where to write trajectory + verifier outputs. Defaults to "
            f"{_default_jobs_dir()}/outside-agent-<task>-<timestamp>/."
        ),
    )
    p.add_argument(
        "--env-file", default=None,
        help=(
            "Optional .env file with KEY=VALUE secrets (API keys, etc.). "
            "Each key is set as a container env var via "
            "acquire_container(environment=...). Operator-side parse — "
            "no shell sourcing required. See xrlenv.client.dotenv."
        ),
    )
    p.add_argument(
        "--upload-env-file", default=None,
        help=(
            "Optional .env file to copy verbatim into the container as a "
            "file (default in-container path /workspace/.env). For "
            "in-container tools that auto-load dotenv vs reading env "
            "vars. Can be combined with --env-file (which sets env "
            "vars) — the two shapes are independent."
        ),
    )
    return p


async def run(args: argparse.Namespace) -> int:
    host = args.host or os.environ.get("XRLENV_GRPC_HOST", "127.0.0.1")
    port = args.port or int(os.environ.get("XRLENV_GRPC_PORT", "50051"))
    token = os.environ.get("XRLENV_CONSUMER_TOKEN") or None

    task = _load_task_bundle(args.task)
    # Artifact directory: per-run, gitignored under <repo>/tmp/ by
    # default. Created up front so the trajectory writer can stream
    # into it as actions land (debuggable even on a partial run).
    artifacts_dir = (
        Path(args.save_artifacts).expanduser()
        if args.save_artifacts
        else _default_jobs_dir() / _default_run_id(task.task_id)
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "task.json").write_text(
        json.dumps({
            "task_id": task.task_id,
            "task_dir": str(task.task_dir),
            "docker_image": task.docker_image,
            "instruction": task.instruction,
        }, indent=2),
        encoding="utf-8",
    )
    (artifacts_dir / "instruction.md").write_text(
        task.instruction + "\n", encoding="utf-8",
    )

    print(
        f"[outside-agent] task={task.task_id!r}\n"
        f"  image: {task.docker_image}\n"
        f"  instruction: {task.instruction[:120]}"
        f"{'…' if len(task.instruction) > 120 else ''}\n"
        f"  artifacts: {artifacts_dir}",
        flush=True,
    )
    print(
        f"[outside-agent] dialing control plane at {host}:{port}",
        flush=True,
    )

    # Parse --env-file BEFORE acquire so secrets land as container
    # env vars at creation time (rather than via a post-acquire exec
    # call that would briefly leave the agent without them).
    env_from_file: dict[str, str] | None = None
    if args.env_file:
        env_from_file = parse_dotenv(args.env_file)
        print(
            f"[outside-agent] loaded {len(env_from_file)} env var(s) from "
            f"{args.env_file} for container creation",
            flush=True,
        )

    client = Client.grpc(host=host, port=port, token=token)
    try:
        async with await client.acquire_container(
            image=task.docker_image,
            command=["sleep", "infinity"],
            environment=env_from_file,
            labels={
                "workflow": "agent-outside-demo",
                "task": task.task_id,
            },
        ) as session:
            print(
                f"[outside-agent] container ready "
                f"(rollout_id={session.rollout_id} node={session.node_id})",
                flush=True,
            )

            # Optionally also copy the .env file in verbatim, for
            # in-container tools that read a file (vs env vars).
            if args.upload_env_file:
                landed_at = await upload_dotenv(
                    session, source=args.upload_env_file,
                )
                print(
                    f"[outside-agent] copied .env into container at "
                    f"{landed_at}",
                    flush=True,
                )

            agent = OracleAgent(solve_script=task.solve_script)
            obs = Observation(
                step=0, last_action=None,
                exit_code=0, stdout="", stderr="",
            )
            trajectory: list[tuple[Action, Observation]] = []
            trajectory_path = artifacts_dir / "trajectory.jsonl"
            start_wall_clock = time.monotonic()
            while True:
                action = agent.next_action(obs)
                if action is None:
                    print("[outside-agent] agent signalled done", flush=True)
                    break

                preview = action.bash.strip().splitlines()[0][:80] + "…"
                print(
                    f"[step {obs.step + 1}] running oracle action "
                    f"(first line: {preview!r})",
                    flush=True,
                )
                result = await session.exec(
                    ["bash", "-c", action.bash],
                    timeout_s=300,
                )
                obs = Observation(
                    step=obs.step + 1,
                    last_action=action.bash,
                    exit_code=result.exit_code,
                    stdout=result.stdout.decode(errors="replace").rstrip(),
                    stderr=result.stderr.decode(errors="replace").rstrip(),
                )
                trajectory.append((action, obs))
                # Stream trajectory to disk so a partial run still
                # leaves debuggable artifacts on the operator's host.
                with trajectory_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "step": obs.step,
                        "action": asdict(action),
                        "observation": asdict(obs),
                    }) + "\n")
                print(
                    f"[step {obs.step}] exit={obs.exit_code} "
                    f"stdout={(obs.stdout[:160] + '…') if len(obs.stdout) > 160 else obs.stdout!r}",
                    flush=True,
                )
                if obs.exit_code != 0:
                    print(
                        f"[outside-agent] action failed; aborting "
                        f"loop. stderr={obs.stderr[:400]!r}",
                        flush=True,
                    )
                    break

            # ── Verifier — same contract harbor's runner uses ────────────
            print(
                f"[outside-agent] running verifier "
                f"(timeout {args.verifier_timeout_s:.0f}s)",
                flush=True,
            )
            # Docker's put_archive refuses to create the target dir
            # itself — it only extracts INTO an existing path. Mkdir
            # both target trees up front (cheap + idempotent).
            await session.exec(
                ["mkdir", "-p", "/tests", "/logs/verifier"],
                timeout_s=30,
            )
            tests_tarball = _tar_dir_contents(task.tests_dir)
            await session.put_archive(
                target_dir="/tests", tarball=tests_tarball,
            )
            verifier_result = await session.exec(
                ["bash", "/tests/test.sh"],
                timeout_s=args.verifier_timeout_s,
            )
            # Pull /logs/verifier/ back to the operator's disk —
            # reward.txt + ctrf.json + anything else the verifier
            # wrote. Same dir layout harbor archives per trial.
            try:
                verifier_tarball = await session.get_archive(
                    "/logs/verifier",
                )
                _untar_to_dir(verifier_tarball, artifacts_dir)
            except Exception as exc:
                print(
                    f"[outside-agent] WARN: couldn't pull /logs/verifier: "
                    f"{exc}",
                    flush=True,
                )
            (artifacts_dir / "verifier_stdout.log").write_bytes(
                verifier_result.stdout,
            )
            (artifacts_dir / "verifier_stderr.log").write_bytes(
                verifier_result.stderr,
            )
            reward_read = await session.exec(
                ["cat", "/logs/verifier/reward.txt"],
                timeout_s=10,
            )
            reward_str = reward_read.stdout.decode(errors="replace").strip()
            passed = (
                reward_read.exit_code == 0
                and reward_str == "1"
            )
            duration_s = time.monotonic() - start_wall_clock
            (artifacts_dir / "summary.json").write_text(
                json.dumps({
                    "task_id": task.task_id,
                    "passed": passed,
                    "reward": reward_str,
                    "verifier_exit_code": verifier_result.exit_code,
                    "step_count": len(trajectory),
                    "duration_s": round(duration_s, 3),
                }, indent=2),
                encoding="utf-8",
            )
            print(
                f"\n[outside-agent] verifier: exit={verifier_result.exit_code} "
                f"reward={reward_str!r} → "
                f"{'PASS' if passed else 'FAIL'}",
                flush=True,
            )
            print(
                f"[outside-agent] artifacts written to {artifacts_dir}",
                flush=True,
            )
            if not passed:
                # Surface the last 600 chars of verifier output so the
                # operator can diagnose without a separate log fetch.
                tail = verifier_result.stdout.decode(
                    errors="replace",
                )[-600:]
                print(
                    f"[outside-agent] verifier stdout tail:\n{tail}",
                    flush=True,
                )
            return 0 if passed else 1
    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(_build_parser().parse_args())))
