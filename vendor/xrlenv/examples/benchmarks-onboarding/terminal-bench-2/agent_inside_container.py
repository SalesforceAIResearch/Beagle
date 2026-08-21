"""Pattern 2: inside-the-container agent (installed agent).

A "real" coding agent — here ``claude-code`` — runs **inside** the
container. The benchmark harness (harbor) drives the install (npm /
binary download) using its own logic, hands the agent the task
instruction, and waits for the agent to finish autonomously. xrlenv
provides the container; everything else lives in harbor.

The script wires the same plumbing as ``smoke.py`` but with two
overrides:

1. **AgentConfig.name = "claude-code"** — harbor resolves this to
   ``harbor.agents.installed.claude_code:ClaudeCode``, which knows
   how to install itself in the container and run.
2. **One task only** — the example optimizes for readability; pass
   ``--task <id>`` to pick a different one from
   ``SMOKE_TASKS`` (defined in ``smoke.py``).

When to use this pattern
========================

- The agent you want to evaluate is a real installable CLI
  (Claude Code, Aider, Codex, OpenHands, etc.) that runs inside the
  task's container.
- You want the agent to drive the workspace autonomously, then have
  the benchmark verifier judge the result. (As opposed to:
  Pattern 1, where your training stack decides each action.)
- You don't want to write any install logic. Harbor's installed
  agents already ship that.

Prerequisites
=============

This example talks to a real LLM through ``claude-code``, so you
need an API key. Either:

- ``ANTHROPIC_API_KEY=...`` for Anthropic models (the default).
- ``OPENAI_API_KEY=...`` if you're pointing claude-code at an
  OpenAI-compatible endpoint.
- A vendor-specific key consumed by whichever model the agent's
  ``model_name`` ultimately resolves to.

The example refuses to start if no API key is present. See the
**Fail-fast** section below.

Operator setup is identical to ``smoke.py``: a control plane
reachable at ``XRLENV_GRPC_HOST:XRLENV_GRPC_PORT``, at least one
node attached, and the task's image
(``hb__<task_id>`` or the upstream ``alexgshaw/<task>:<rev>``
mirror) available on that node.

Run with::

    export ANTHROPIC_API_KEY=sk-ant-...
    export XRLENV_GRPC_HOST=127.0.0.1
    .venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/agent_inside_container.py \\
        --task fix-git
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import logging
import os
import pprint
import sys
from pathlib import Path
from typing import Any

# Reuse the same task-locator helpers as the benchmark smoke so
# both onboarding paths consume the same harbor cache layout.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke import (
    CLUSTER_IMPORT_PATH,
    LOCAL_IMPORT_PATH,
    SMOKE_TASKS,
    _default_jobs_dir,
    _locate_task_dir,
)

LOGGER = logging.getLogger("xrlenv.examples.agent_inside_container")


# ──────────────────────────────────────────────────────────────────────────────
# Fail-fast on missing API credentials.
# ──────────────────────────────────────────────────────────────────────────────

# Recognized provider keys, in priority order. The example exits 2
# if NONE of these are present in the environment. The actual key
# the agent ends up using depends on which model ``--model``
# resolves to; we deliberately don't filter by the chosen model
# because users often have multiple keys present and pick at runtime.
_CREDENTIAL_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)


def _check_credentials() -> None:
    found = [v for v in _CREDENTIAL_ENV_VARS if os.environ.get(v)]
    if not found:
        raise SystemExit(
            "no LLM API key found in the environment. claude-code needs "
            "credentials to call a model. Set one of: "
            f"{', '.join(_CREDENTIAL_ENV_VARS)}.\n"
            "Example: export ANTHROPIC_API_KEY=sk-ant-...",
        )
    LOGGER.info("credentials detected: %s", found)


# ──────────────────────────────────────────────────────────────────────────────
# Job composition.
# ──────────────────────────────────────────────────────────────────────────────


def _default_job_id() -> str:
    return _dt.datetime.utcnow().strftime(
        "claude-code-tb2-%Y%m%d-%H%M%S",
    )


def _build_job_config(
    *,
    task_id: str,
    local: bool,
    jobs_dir: Path,
    job_id: str,
    model_name: str | None,
    max_turns: int | None,
    timeout_sec: float | None,
) -> Any:
    """Compose harbor's ``JobConfig`` with claude-code as the agent.

    The only differences from ``smoke.py``'s helper are:
    - ``AgentConfig(name="claude-code", ...)`` instead of the
      default ``AgentConfig()`` (which resolves to harbor's
      OracleAgent).
    - One task in the ``tasks=[...]`` list (this example is for
      readability; smoke.py runs many).
    - Optional ``override_timeout_sec`` to give a long-form agent
      enough time to actually finish.
    """
    from harbor.models.job.config import JobConfig
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
    )

    import_path = LOCAL_IMPORT_PATH if local else CLUSTER_IMPORT_PATH

    agent_kwargs: dict[str, Any] = {}
    if max_turns is not None:
        agent_kwargs["max_turns"] = max_turns

    return JobConfig(
        job_name=job_id,
        jobs_dir=jobs_dir,
        n_concurrent_trials=1,
        environment=EnvironmentConfig(import_path=import_path),
        agents=[
            AgentConfig(
                name="claude-code",
                model_name=model_name,
                override_timeout_sec=timeout_sec,
                kwargs=agent_kwargs,
            ),
        ],
        tasks=[TaskConfig(path=_locate_task_dir(task_id))],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Result printing.
# ──────────────────────────────────────────────────────────────────────────────


def _print_trial_result(trial_result: Any) -> int:
    print("\n=== claude-code on terminal-bench-2 ===")
    if trial_result.exception_info is not None:
        print(
            f"FAIL  task={trial_result.task_name} "
            f"exception={trial_result.exception_info.exception_type}",
        )
        return 1
    vr = trial_result.verifier_result
    if vr is None or vr.rewards is None or not vr.rewards:
        print(
            f"FAIL  task={trial_result.task_name} reason='no verifier "
            f"rewards recorded'",
        )
        return 1
    failures = [k for k, v in vr.rewards.items() if not (v > 0)]
    if failures:
        print(
            f"FAIL  task={trial_result.task_name} "
            f"non-positive rewards: {failures}",
        )
        pprint.pp(dict(vr.rewards))
        return 1
    print(f"PASS  task={trial_result.task_name}")
    pprint.pp(dict(vr.rewards))
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Entry point.
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent_inside_container",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task", default=SMOKE_TASKS[0],
        help=(
            f"terminal-bench-2 task id to run "
            f"(default: {SMOKE_TASKS[0]!r}). Pass any task id "
            f"present in your harbor cache."
        ),
    )
    p.add_argument(
        "--local", action="store_true",
        help=(
            "Run against the local Docker daemon via harbor's stock "
            "DockerEnvironment wrapped by xrlenv. Default: cluster "
            "mode via the xrlenv control plane."
        ),
    )
    p.add_argument(
        "--model", default=None,
        help=(
            "Override the model claude-code uses (e.g. "
            "'claude-sonnet-4-5'). Defaults to whatever claude-code "
            "picks from $ANTHROPIC_MODEL / its built-in default."
        ),
    )
    p.add_argument(
        "--max-turns", type=int, default=None,
        help="Cap on claude-code's interaction turns. Default: agent default.",
    )
    p.add_argument(
        "--timeout-sec", type=float, default=None,
        help=(
            "Override the per-trial timeout in seconds. Default: the "
            "task's declared timeout (typically 1200s for TB-2 tasks)."
        ),
    )
    p.add_argument(
        "--save-artifacts", default=None,
        help=(
            "Where to write harbor's per-trial outputs. Defaults to "
            "<repo>/tmp/ (gitignored)."
        ),
    )
    return p


async def _run(args: argparse.Namespace) -> int:
    import harbor

    _check_credentials()

    if not args.local:
        if not os.environ.get("XRLENV_GRPC_HOST"):
            raise SystemExit(
                "cluster mode: XRLENV_GRPC_HOST not set. Either pass "
                "--local for the local-Docker baseline, or export "
                "XRLENV_GRPC_HOST / XRLENV_GRPC_PORT / "
                "XRLENV_CONSUMER_TOKEN before running.",
            )
        LOGGER.info(
            "cluster mode: control plane at %s:%s",
            os.environ.get("XRLENV_GRPC_HOST"),
            os.environ.get("XRLENV_GRPC_PORT", "50051"),
        )

    jobs_dir = (
        Path(args.save_artifacts).expanduser()
        if args.save_artifacts else _default_jobs_dir()
    )
    job_id = _default_job_id()
    jobs_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "running task=%s with agent=claude-code (model=%s, max_turns=%s)",
        args.task, args.model, args.max_turns,
    )
    LOGGER.info("artifacts: %s", jobs_dir / job_id)

    config = _build_job_config(
        task_id=args.task,
        local=args.local,
        jobs_dir=jobs_dir,
        job_id=job_id,
        model_name=args.model,
        max_turns=args.max_turns,
        timeout_sec=args.timeout_sec,
    )

    job = await harbor.Job.create(config)
    job_result = await job.run()

    trial_results = list(job_result.trial_results)
    if not trial_results:
        print("FAIL  harbor returned zero trial results (job aborted)")
        return 1
    exit_code = _print_trial_result(trial_results[0])
    print(f"artifacts at {jobs_dir / job_id}")
    return exit_code


def main() -> int:
    args = _build_parser().parse_args()
    from xrlenv.observability.logging import configure_logging
    configure_logging()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
