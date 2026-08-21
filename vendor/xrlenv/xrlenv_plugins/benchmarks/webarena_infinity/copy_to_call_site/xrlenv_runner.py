#!/usr/bin/env python3
"""In-container single-task runner for the xrlenv evaluation path.

This runs *inside* an xrlenv webarena-infinity container. The host orchestrator
(``run_eval_parallel_xrlenv.py``) cannot reach the app server's port across the
container boundary, so the browser-use agent and the verifier must run here,
next to the server on ``localhost``.

It is deliberately split into two phases so the answer-free contract holds at
runtime (audit H1/D6):

  phase ``agent``   reset seed state, drive the browser-use agent, save the
                    trajectory (history.json + screenshots) and ``_agent.json``
                    (agent metrics only — no pass/fail). NO verifier on disk.
  phase ``verify``  load the verifier the host injected *after* the agent
                    exited, grade the live server state, write ``result.json``
                    (the full per-task result dict, same shape the local
                    ``tasks.run_task`` produces).

The app server is started once per container by the orchestrator (backgrounded,
persistent) and reset between tasks — this runner only connects to it. The task
spec is injected by the host as a single-task JSON file (``--task-json``) so the
run is identical regardless of the pinned image's bundled ``*-tasks.json``.

Invoked by the orchestrator as, e.g.::

    cd /opt/webarena-infinity && python3 evaluation/xrlenv_runner.py \
        --phase agent --model gemini-pro --web-app apps/gmail \
        --task-json /work/<task>/task.json --out /work/<task> \
        --server-url http://localhost:9000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# This file lives in evaluation/; make its siblings importable when invoked by
# absolute path (cwd is the repo root, not evaluation/).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import AgentResult  # noqa: E402
from run_eval_parallel import AGENT_FACTORIES  # noqa: E402
from tasks import TASK_TIMEOUT, load_verifier, reset_state  # noqa: E402


def _load_task(task_json: str) -> dict:
    with open(task_json) as f:
        return json.load(f)


def _history_step_count(task_dir: Path) -> int:
    """Best-effort step count from a saved history.json (for timeouts)."""
    hist = task_dir / "history.json"
    if not hist.exists():
        return -1
    try:
        with open(hist) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return -1
    return len(data) if isinstance(data, list) else len(data.get("history", []))


def _base_meta(task: dict) -> dict:
    return {
        "task_id": task["id"],
        "difficulty": task.get("difficulty", ""),
        "instruction": task["instruction"],
        "status": "ok",          # ok | timeout | error
        "elapsed": 0,
        "steps": -1,
        "is_done": False,
        "final_result": None,
        "errors": [],
    }


async def _run_browser_agent(args: argparse.Namespace, task: dict, task_dir: Path) -> dict:
    """The real agent: browser-use drives the app over CDP."""
    agent = AGENT_FACTORIES[args.model](
        use_vision=args.use_vision,
        max_steps=args.max_steps,
        timeout=TASK_TIMEOUT,
        headless=True,
    )
    meta = _base_meta(task)
    try:
        await agent.setup(args.server_url)
        # Restore seed state before the agent acts (mirrors tasks.run_task).
        reset_state(args.server_url)
        try:
            result: AgentResult = await agent.run(
                task=task["instruction"],
                server_url=args.server_url,
                task_dir=task_dir,
            )
            meta.update(
                status="ok",
                elapsed=result.elapsed,
                steps=result.steps,
                is_done=result.is_done,
                final_result=result.final_result,
                errors=result.errors,
            )
        except asyncio.TimeoutError:
            # history.json + screenshots are saved by the agent before re-raise.
            meta.update(
                status="timeout",
                elapsed=TASK_TIMEOUT,
                steps=_history_step_count(task_dir),
                errors=[f"Timeout after {TASK_TIMEOUT}s"],
            )
    except Exception as e:  # browser setup failure or unexpected agent crash
        meta.update(status="error", errors=[str(e)])
    finally:
        try:
            await agent.teardown()
        except Exception:
            pass
    return meta


# Per-app sanity_check files are inconsistent; the oracle tolerates all variants.
_SEED_FN_NAMES = ("load_seed_state", "generate_seed_state", "get_seed_state")


def _oracle_solver_file(app_dir: Path, suite: str) -> str | None:
    """Resolve the sanity_check solver filename for a suite (names vary by app:
    real → sanity_check_real.py or older sanity_check.py; function →
    sanity_check_function.py). Returns None if none is present. Shared with the
    orchestrator so host-injection and in-container import pick the same file.
    """
    candidates = (
        ["sanity_check_function.py"] if suite == "function-tasks"
        else ["sanity_check_real.py", "sanity_check.py"]
    )
    return next((c for c in candidates if (app_dir / c).exists()), None)


def _run_oracle(args: argparse.Namespace, task: dict) -> dict:
    """The 'oracle agent': set the env to the solved state directly (no browser).

    It plugs into the exact same per-task lifecycle as the browser agent — reset
    the *reused* server to seed, then act — so a multi-task run exercises the
    orchestrator's server-reuse + reset-between-tasks path. The solution comes
    from the app's sanity_check solver (injected by the orchestrator in oracle
    mode); no LLM, no Chromium. Tolerates per-app naming drift; on missing
    solution it returns a per-task error rather than crashing the worker.
    """
    import importlib.util

    import requests

    suite = getattr(args, "task_suite", "real-tasks")
    meta = _base_meta(task)
    meta["steps"] = 1

    app_dir = Path(args.web_app)
    solver_file = _oracle_solver_file(app_dir, suite)
    if solver_file is None:
        meta.update(status="error", errors=[f"no sanity_check solver for {suite} in {args.web_app}"])
        return meta
    try:
        spec = importlib.util.spec_from_file_location("oracle_solver", app_dir / solver_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Solver dispatch varies: a SOLVERS dict, or bare solve_<task_id> funcs.
        solvers = getattr(mod, "SOLVERS", None) or {}
        solver = solvers.get(task["id"]) or getattr(mod, f"solve_{task['id']}", None)
        if solver is None:
            meta.update(status="error", errors=[f"no oracle solution for {task['id']}"])
            return meta

        # No browser to capture seed: PUT the Node-computed seed on first use so
        # the server has a _seed_state to reset to (idempotent per container).
        # Seeder name varies (load_/generate_/get_seed_state).
        if requests.get(f"{args.server_url}/api/state", timeout=5).status_code != 200:
            seed_fn = next((getattr(mod, n) for n in _SEED_FN_NAMES if hasattr(mod, n)), None)
            if seed_fn is None:
                meta.update(status="error", errors=["sanity_check has no seed-state function"])
                return meta
            requests.put(f"{args.server_url}/api/state", json=seed_fn(), timeout=15)
        reset_state(args.server_url)                          # restore seed (every task)
        state = requests.get(f"{args.server_url}/api/state", timeout=5).json()
        solver(state)                                         # apply the oracle solution
        requests.put(f"{args.server_url}/api/state", json=state, timeout=15)
        meta["is_done"] = True   # the oracle always completes its single action
    except Exception as e:
        meta.update(status="error", errors=[str(e)])
    return meta


async def run_agent_phase(args: argparse.Namespace) -> int:
    """Phase A: act on the (reused) server, save agent metrics. No verifier."""
    task = _load_task(args.task_json)
    task_dir = Path(args.out)
    task_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "oracle":
        meta = _run_oracle(args, task)
    else:
        meta = await _run_browser_agent(args, task, task_dir)

    with open(task_dir / "_agent.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"[xrlenv-runner] agent phase ({args.model}): status={meta['status']} "
        f"steps={meta['steps']} elapsed={meta['elapsed']}s",
        flush=True,
    )
    return 0


def run_verify_phase(args: argparse.Namespace) -> int:
    """Phase B: grade the live server state with the injected verifier."""
    task = _load_task(args.task_json)
    task_dir = Path(args.out)

    meta_path = task_dir / "_agent.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}

    try:
        verify_fn = load_verifier(args.web_app, task["verify"])
        passed, verifier_message = verify_fn(args.server_url)
    except Exception as e:
        passed, verifier_message = False, f"Verifier exception: {e}"

    result = {
        "task_id": task["id"],
        "difficulty": task.get("difficulty", ""),
        "instruction": task["instruction"],
        "passed": bool(passed),
        "verifier_message": verifier_message,
        "elapsed": meta.get("elapsed", 0),
        "steps": meta.get("steps", -1),
        "is_done": meta.get("is_done", False),
        "final_result": meta.get("final_result"),
        "errors": meta.get("errors", []),
    }
    with open(task_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[xrlenv-runner] verify phase: passed={passed} :: {verifier_message}",
        flush=True,
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="In-container single-task runner (xrlenv path)"
    )
    p.add_argument("--phase", choices=["agent", "verify"], required=True)
    p.add_argument("--model", choices=list(AGENT_FACTORIES.keys()) + ["oracle"], default="gpt")
    p.add_argument("--web-app", required=True,
                   help="App dir relative to CWD, e.g. apps/gmail")
    p.add_argument("--task-suite", default="real-tasks",
                   help="Suite name (selects the oracle solver in oracle mode)")
    p.add_argument("--task-json", required=True,
                   help="Path to a single task's JSON (host-injected)")
    p.add_argument("--out", required=True,
                   help="Output dir for this task's artifacts")
    p.add_argument("--server-url", default="http://localhost:9000",
                   help="URL of the app server inside the container")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--use-vision", action="store_true")
    args = p.parse_args()

    if args.phase == "agent":
        return asyncio.run(run_agent_phase(args))
    return run_verify_phase(args)


if __name__ == "__main__":
    sys.exit(main())
