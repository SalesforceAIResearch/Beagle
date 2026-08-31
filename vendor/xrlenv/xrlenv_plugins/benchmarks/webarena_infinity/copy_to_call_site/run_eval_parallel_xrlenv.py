#!/usr/bin/env python3
"""Parallel evaluation runner that distributes work to the xrlenv cluster.

Counterpart to ``run_eval_parallel.py``. Same CLI surface and same output
layout (results.json + report.html, multi-run merge, resume), but instead of
starting an app server on a local host port, each worker drives a container on
the xrlenv cluster.

Why it's shaped differently from the local runner
-------------------------------------------------
xrlenv hands you a container with **no reachable port**. So the app server, the
browser, the browser-use agent, and the verifier all run *inside* the container
(everything talks to ``localhost`` there). This host process only orchestrates:

  • acquire a container per worker (reused across many tasks)
  • inject ``evaluation/`` (incl. ``xrlenv_runner.py``) once per container
  • start the app server inside the container once (backgrounded, persistent)
  • per task: run the agent (phase A) → inject the verifier (the answer, only
    now) → run the verifier (phase B) → delete the answer → pull artifacts back
  • aggregate + report on the host (shared helpers from run_eval_parallel.py)

The verifier is injected only *after* the agent has exited and removed before
the next task, so the answer-free image (audit H1/D6) stays answer-free at
runtime — the agent is never co-resident with an answer file.

Environment / prerequisites
---------------------------
  • The substrate image must be built+pushed (see
    xrlenv_plugins/benchmarks/webarena_infinity/README.md).
  • ``xrlenv`` must be importable (set XRLENV_REPO or pip install it).
  • All config lives in the project-root ``.env`` (read once by xrlenv_config):
      XRLENV_GRPC_HOST, XRLENV_GRPC_PORT, XRLENV_CONSUMER_TOKEN,
      XRLENV_PRIVATE_REGISTRY_HOST, XRLENV_PRIVATE_REGISTRY_PORT,
      and LLM API keys (OPENAI_API_KEY / GOOGLE_API_KEY / GEMINI_API_KEY /
      ANTHROPIC_API_KEY) — the keys are forwarded into each container.

Usage
-----
    # 8 containers, real-tasks suite, against the dev control plane
    python evaluation/run_eval_parallel_xrlenv.py --model gemini-pro --workers 8 \
        --web-app apps/gmail

    # one task, explicit image + control plane
    python evaluation/run_eval_parallel_xrlenv.py --model gpt --task-id task_e1 \
        --workers 1 --web-app apps/gmail \
        --image <registry-host>:5011/xrlenv-webarena-infinity/substrate:dev \
        --xrlenv-host <control-plane-host> --xrlenv-port 50051
"""

import argparse
import asyncio
import io
import json
import os
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path

# Make sibling evaluation/ modules importable when run by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report import generate_report  # noqa: E402
from tasks import TASK_TIMEOUT, filter_tasks, load_tasks  # noqa: E402
from xrlenv_runner import _oracle_solver_file  # noqa: E402  (shared solver-name resolution)

# Reuse the local runner's model registry, multi-run helpers, and colors so the
# two paths can never drift. Importing it is cheap on the host: the browser-use
# imports inside AGENT_FACTORIES are lazy (only fire inside the container).
from run_eval_parallel import (  # noqa: E402
    AGENT_FACTORIES,
    BG_GREEN,
    BG_RED,
    BOLD,
    CYAN,
    DIFF_COLOR,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    RESET,
    WHITE,
    YELLOW,
    add_monet_args,
    apply_monet_overrides,
    find_incomplete_tasks,
    merge_repetition_results,
)

# All cluster coordinates + credentials come from the project-root .env, read in
# exactly one place: xrlenv_config (which also performs the guarded xrlenv import
# and loads .env at its import). Nothing here touches os.environ for config.
from xrlenv_config import (  # noqa: E402
    MONET_CONTAINER_REPO,
    MONET_GIT_REF,
    MONET_GIT_URL,
    default_image,
    llm_env,
    make_client,
    monet_env,
    monet_preflight,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTAINER_REPO = "/opt/webarena-infinity"   # where the image clones the repo
INTERNAL_PORT = 9000                        # app server port inside container
WORKDIR = "/work"                           # per-task scratch inside container


# ---------------------------------------------------------------------------
# Tar / file-transfer helpers
# ---------------------------------------------------------------------------


def _tar_dir_contents(src: Path) -> bytes:
    """Tar the *contents* of src (children land directly under target_dir)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for entry in sorted(src.iterdir()):
            tf.add(entry, arcname=entry.name)
    return buf.getvalue()


def _tar_one_file(src: Path, arcname: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        tf.add(src, arcname=arcname)
    return buf.getvalue()


# App code is injected from the host working tree (not the pinned image) so the
# app, its *-tasks.json, and its verifiers all come from the same source. The
# answer dirs/solvers are excluded — verifiers are injected per task, post-agent.
_APP_EXCLUDE_DIRS = {"real-tasks", "function-tasks", "results", "__pycache__"}


def _tar_app_contents(app_dir: Path) -> bytes:
    def _filt(ti: tarfile.TarInfo):
        parts = Path(ti.name).parts
        if any(p in _APP_EXCLUDE_DIRS for p in parts):
            return None
        base = Path(ti.name).name
        if base.startswith("sanity_check_") and base.endswith(".py"):
            return None
        return ti

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for entry in sorted(app_dir.iterdir()):
            if entry.name in _APP_EXCLUDE_DIRS:
                continue
            tf.add(entry, arcname=entry.name, filter=_filt)
    return buf.getvalue()


def _tar_bytes(name: str, data: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _extract_into(tarball: bytes, dest_parent: Path) -> None:
    dest_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r") as tf:
        tf.extractall(dest_parent, filter="data")


async def _mkdir(session, *dirs: str) -> None:
    await session.exec(["mkdir", "-p", *dirs], timeout_s=30)


async def _put_dir(session, local_dir: Path, target_dir: str) -> None:
    await _mkdir(session, target_dir)
    await session.put_archive(target_dir=target_dir, tarball=_tar_dir_contents(local_dir))


async def _put_file(session, local_file: Path, target_dir: str, arcname: str | None = None) -> None:
    await _mkdir(session, target_dir)
    await session.put_archive(
        target_dir=target_dir,
        tarball=_tar_one_file(local_file, arcname or local_file.name),
    )


async def _put_text(session, target_dir: str, name: str, text: str) -> None:
    await _mkdir(session, target_dir)
    await session.put_archive(target_dir=target_dir, tarball=_tar_bytes(name, text.encode()))


async def _read_json(session, path: str):
    r = await session.exec(["cat", path], timeout_s=30)
    if r.exit_code != 0:
        return None
    try:
        return json.loads(r.stdout.decode())
    except json.JSONDecodeError:
        return None


async def _pull_dir(session, container_dir: str, dest_parent: Path) -> None:
    """get_archive(container_dir) → dest_parent/<basename>/..."""
    try:
        tarball = await session.get_archive(container_dir)
    except Exception:
        return
    _extract_into(tarball, dest_parent)


# ---------------------------------------------------------------------------
# In-container app server lifecycle
# ---------------------------------------------------------------------------


async def _start_app_server(session, web_app_rel: str, port: int) -> None:
    """Start the app server backgrounded so it outlives this exec call.

    setsid + nohup detach the server into its own session; the exec returns
    immediately while the server keeps running for the container's lifetime.
    The app's server.py serves files relative to CWD, so we cd into it first.
    """
    app_dir = f"{CONTAINER_REPO}/{web_app_rel}"
    launch = (
        f"cd {app_dir} && setsid nohup python3 server.py --port {port} "
        f"> {WORKDIR}/server.log 2>&1 < /dev/null & echo started"
    )
    await session.exec(["bash", "-lc", launch], timeout_s=30)


async def _wait_app_server(session, port: int, tries: int = 40) -> bool:
    poll = (
        f"for i in $(seq 1 {tries}); do "
        f"curl -sf -o /dev/null http://localhost:{port}/ && exit 0; sleep 0.5; "
        f"done; exit 1"
    )
    r = await session.exec(["bash", "-lc", poll], timeout_s=tries + 10)
    return r.exit_code == 0


async def _provision_container(
    session, eval_dir: str, web_app_rel: str, web_app_dir: str, oracle_suite: str | None = None
) -> None:
    """One-time per-container setup: inject evaluation/ + host app code (minus
    answers), then start the app server. Raises on failure (incl. server not up).
    Shared by the orchestrator's worker and the smoke tests.

    In oracle mode (``oracle_suite`` set) the app's sanity_check solver — normally
    stripped as an answer — is also injected, since the oracle *is* the solver.
    """
    await _put_dir(session, Path(eval_dir), f"{CONTAINER_REPO}/evaluation")
    app_target = f"{CONTAINER_REPO}/{web_app_rel}"
    await _mkdir(session, app_target)
    await session.put_archive(target_dir=app_target, tarball=_tar_app_contents(Path(web_app_dir)))
    await _mkdir(session, WORKDIR)  # verifier dir (real-tasks/ or function-tasks/) is created on injection
    if oracle_suite:
        # Solver filename varies per app; inject whichever exists. If none does,
        # skip — the oracle phase reports a per-task error rather than crashing.
        solver = _oracle_solver_file(Path(web_app_dir), oracle_suite)
        if solver:
            await _put_file(session, Path(web_app_dir) / solver, app_target, arcname=solver)
    await _start_app_server(session, web_app_rel, INTERNAL_PORT)
    if not await _wait_app_server(session, INTERNAL_PORT):
        log = await session.exec(["tail", "-c", "1000", f"{WORKDIR}/server.log"], timeout_s=10)
        raise RuntimeError(
            f"app server failed to start:\n{log.stdout.decode(errors='replace')}"
        )


# ---------------------------------------------------------------------------
# Monet install (--model monet): clone from GitHub + npm ci, once per container
# ---------------------------------------------------------------------------


async def _install_monet(session, tag: str) -> None:
    """Clone Monet into MONET_CONTAINER_REPO and `npm ci`, once per container.

    The repo is private, so the clone authenticates over HTTPS with $GH_TOKEN —
    expanded by the container's shell (the literal "$GH_TOKEN" is what we send, so
    the token VALUE never appears in the host-side argv). Node + git ship in the
    substrate image; `npm ci` uses Monet's committed lockfile. Idempotent: skips
    the clone if a prior call on this container already populated the dir.
    """
    clone_url = f"https://$GH_TOKEN@{MONET_GIT_URL}"  # $GH_TOKEN expands IN-container
    script = (
        "set -e; "
        f"if [ ! -e {MONET_CONTAINER_REPO}/bin/monet.js ]; then "
        f'  git clone --depth 1 --branch {MONET_GIT_REF} "{clone_url}" {MONET_CONTAINER_REPO}; '
        f"  cd {MONET_CONTAINER_REPO} && npm ci; "
        "fi; "
        f"node {MONET_CONTAINER_REPO}/bin/monet.js --version"
    )
    r = await session.exec(["bash", "-lc", script], timeout_s=900)
    if r.exit_code != 0:
        err = r.stderr.decode(errors="replace")[-1000:]
        raise RuntimeError(f"monet install failed (exit {r.exit_code}):\n{err}")
    ver = (r.stdout.decode(errors="replace").strip().splitlines() or ["installed"])[-1]
    print(f"  {tag} monet ready @ {MONET_CONTAINER_REPO} ({ver})")


# ---------------------------------------------------------------------------
# Result synthesis (non-ok agent phases — mirrors local worker behavior)
# ---------------------------------------------------------------------------


def _synth_result(task: dict, meta: dict, message: str | None = None) -> dict:
    status = meta.get("status", "error")
    if status == "timeout":
        vm = f"Timed out after {TASK_TIMEOUT}s"
    elif status == "error":
        errs = meta.get("errors") or ["Agent crashed"]
        vm = f"Agent error: {' '.join(str(e) for e in errs)}"
    else:
        vm = message or "Unknown failure"
    return {
        "task_id": task["id"],
        "difficulty": task.get("difficulty", ""),
        "instruction": task["instruction"],
        "passed": False,
        "verifier_message": vm,
        "elapsed": meta.get("elapsed", 0),
        "steps": meta.get("steps", -1),
        "is_done": meta.get("is_done", False),
        "final_result": meta.get("final_result"),
        "errors": meta.get("errors", []),
    }


# ---------------------------------------------------------------------------
# Per-task execution inside a container
# ---------------------------------------------------------------------------


async def _run_task_in_container(
    session, task: dict, args, web_app_rel: str, web_app_dir: str, run_dir: Path
) -> dict:
    task_id = task["id"]
    workdir = f"{WORKDIR}/{task_id}"
    await _mkdir(session, workdir)
    await _put_text(session, workdir, "task.json", json.dumps(task))

    vision = "--use-vision" if args.use_vision else ""
    base = (
        f"cd {CONTAINER_REPO} && python3 evaluation/xrlenv_runner.py "
        f"--model {args.model} --web-app {web_app_rel} --task-suite {args.task_suite} "
        f"--task-json {workdir}/task.json --out {workdir} "
        f"--server-url http://localhost:{INTERNAL_PORT}"
    )

    # --- Phase A: agent (no verifier on disk) ---
    agent_cmd = f"{base} --phase agent --max-steps {args.max_steps} {vision}"
    await session.exec(["bash", "-lc", agent_cmd], timeout_s=TASK_TIMEOUT + 180)

    meta = await _read_json(session, f"{workdir}/_agent.json") or {
        "status": "error",
        "errors": ["agent phase produced no _agent.json"],
    }

    if meta.get("status") == "ok":
        # --- Phase B: inject the answer, grade, then delete it ---
        # task["verify"] carries its own subdir ("real-tasks/x.py" or
        # "function-tasks/x.py"), so derive the target dir from it — suite-agnostic.
        verifier_rel = task["verify"]
        verifier_name = Path(verifier_rel).name
        verifier_dir = f"{CONTAINER_REPO}/{web_app_rel}/{Path(verifier_rel).parent}"
        await _put_file(
            session, Path(web_app_dir) / verifier_rel, verifier_dir, arcname=verifier_name
        )
        await session.exec(["bash", "-lc", f"{base} --phase verify"], timeout_s=300)
        await session.exec(["rm", "-f", f"{verifier_dir}/{verifier_name}"], timeout_s=30)
        result = await _read_json(session, f"{workdir}/result.json")
        if result is None:
            result = _synth_result(task, meta, "verify phase produced no result.json")
    else:
        result = _synth_result(task, meta)

    # Pull all artifacts (history.json, screenshots/, result.json, …) to host.
    await _pull_dir(session, workdir, run_dir)
    # Guarantee a result.json on host even if extraction missed it.
    host_task_dir = run_dir / task_id
    host_task_dir.mkdir(parents=True, exist_ok=True)
    with open(host_task_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Container worker (reused across many tasks)
# ---------------------------------------------------------------------------


async def _acquire_eval_container(
    client, image, *, model, app, task_key, group, env_vars, acquire_timeout,
    artifact_path=None, displayed_name=None,
):
    """Acquire a container with dashboard-friendly scheduler metadata.

    - ``xrlenv.group_id`` label: groups every container of one run in the
      dashboard (and makes them cancelable together via client.cancel_group).
    - ``task_key`` MUST be unique per container. It is an *anti-affinity* key:
      the scheduler refuses to place more than ``max_runs_per_task`` (default 4)
      rollouts sharing a task_key on one node. A shared key would cap each node
      at 4 concurrent containers and silently queue the rest of a --workers>4
      run. Unique keys still populate the dashboard's task_key column without
      throttling.
    - ``artifact_path`` / ``displayed_name``: per-rollout admin-dashboard metadata
      passed as ``xrlenv.rollout.*`` labels. The control plane parses these off the
      acquire labels (the same path as ``xrlenv.group_id``) and persists them onto
      the rollout record — without them the dashboard shows "(not set; smoke driver
      didn't pass artifact_path)". (The ``rollout_metadata(...)`` contextvar is NOT
      used here: it only feeds the docker-compat ``containers.create()`` path.)
    """
    labels = {
        "workflow": "webarena-infinity-eval",
        "model": model,
        "app": app,
        "xrlenv.group_id": group,
    }
    if artifact_path:
        labels["xrlenv.rollout.artifact_path"] = artifact_path
    if displayed_name:
        labels["xrlenv.rollout.displayed_name"] = displayed_name
    return await client.acquire_container(
        image=image,
        command=["sleep", "infinity"],
        environment=env_vars or None,
        labels=labels,
        task_key=task_key,
        acquire_timeout_s=acquire_timeout,
    )


async def container_worker(
    worker_id: int,
    task_queue: asyncio.Queue,
    results: list,
    results_lock: asyncio.Lock,
    *,
    client,
    image: str,
    args,
    run_dir: Path,
    web_app_rel: str,
    web_app_dir: str,
    env_vars: dict,
):
    tag = f"{DIM}[C{worker_id}]{RESET}"
    try:
        session_ctx = await _acquire_eval_container(
            client, image,
            model=args.model,
            app=Path(web_app_rel).name,
            task_key=f"{run_dir.name}/c{worker_id}",   # unique per container
            group=run_dir.name,
            env_vars=env_vars,
            acquire_timeout=args.acquire_timeout,
            # where this container's pulled artifacts land + a readable dashboard label
            artifact_path=str(run_dir.resolve()),
            displayed_name=f"c{worker_id} ({Path(web_app_rel).name})",
        )
    except Exception as e:
        print(f"  {tag} {RED}acquire failed: {e}{RESET}")
        return

    async with session_ctx as session:
        print(f"  {tag} container on node={getattr(session, 'node_id', '?')}")
        try:
            oracle_suite = args.task_suite if args.model == "oracle" else None
            await _provision_container(
                session, args.eval_dir, web_app_rel, web_app_dir, oracle_suite=oracle_suite
            )
        except Exception as e:
            print(f"  {tag} {RED}container setup failed: {e}{RESET}")
            return
        if args.model == "monet":
            try:
                await _install_monet(session, tag)
            except Exception as e:
                print(f"  {tag} {RED}monet install failed: {e}{RESET}")
                return
        print(f"  {tag} ready (server :{INTERNAL_PORT})")

        while True:
            try:
                task = task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            diff = task.get("difficulty", "")
            dc = DIFF_COLOR.get(diff, "")
            try:
                result = await _run_task_in_container(
                    session, task, args, web_app_rel, web_app_dir, run_dir
                )
                badge = (
                    f"{BG_GREEN}{WHITE}{BOLD} PASS {RESET}"
                    if result["passed"]
                    else f"{BG_RED}{WHITE}{BOLD} FAIL {RESET}"
                )
                print(
                    f"  {tag} {BOLD}{task['id']}{RESET} {dc}{diff}{RESET}  {badge} "
                    f"{DIM}{result['elapsed']}s  {result['steps']} steps{RESET}"
                )
            except Exception as e:
                print(
                    f"  {tag} {BOLD}{task['id']}{RESET} {dc}{diff}{RESET}  "
                    f"{BG_RED}{WHITE}{BOLD} ERR  {RESET} {DIM}{e}{RESET}"
                )
                result = _synth_result(task, {"status": "error", "errors": [str(e)]})
                host_task_dir = run_dir / task["id"]
                host_task_dir.mkdir(parents=True, exist_ok=True)
                with open(host_task_dir / "result.json", "w") as f:
                    json.dump(result, f, indent=2)

            async with results_lock:
                results.append(result)


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------


def _aggregate_and_report(results: list, args, run_dir: Path, timestamp: str, num_workers: int):
    results.sort(key=lambda r: r["task_id"])
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    by_diff: dict[str, dict] = {}
    for r in results:
        d = r.get("difficulty", "")
        if d:
            by_diff.setdefault(d, {"total": 0, "passed": 0})
            by_diff[d]["total"] += 1
            if r["passed"]:
                by_diff[d]["passed"] += 1

    aggregate = {
        "model": args.model,
        "timestamp": timestamp,
        "use_vision": args.use_vision,
        "workers": num_workers,
        "backend": "xrlenv",
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total * 100, 1) if total else 0,
        "by_difficulty": by_diff,
        "tasks": results,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(aggregate, f, indent=2)
    report_path = generate_report(results, args.model, timestamp, run_dir)
    return aggregate, report_path


async def run_single_eval(
    tasks: list[dict],
    args,
    run_dir: Path,
    web_app_dir: str,
    client,
    image: str,
    env_vars: dict,
    run_label: str = "",
    prior_results: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    num_workers = min(args.workers, len(tasks)) if tasks else 0
    web_app_rel = os.path.relpath(web_app_dir, str(Path(args.repo_root).resolve()))
    timestamp = run_dir.name.split("_", 1)[-1] if "_" in run_dir.name else ""

    resuming = prior_results is not None
    header_label = "  Resuming xrlenv Evaluation" if resuming else "  xrlenv Parallel Evaluation"
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}{header_label}{run_label}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"  {DIM}Model:{RESET}   {BOLD}{args.model}{RESET}")
    print(f"  {DIM}Suite:{RESET}   {BOLD}{args.task_suite}{RESET}")
    print(f"  {DIM}App:{RESET}     {BOLD}{web_app_rel}{RESET}")
    print(f"  {DIM}Tasks:{RESET}   {BOLD}{len(tasks)}{RESET}"
          + (f" {DIM}({len(prior_results)} already done){RESET}" if resuming else ""))
    print(f"  {DIM}Workers:{RESET} {BOLD}{num_workers}{RESET} {DIM}(containers){RESET}")
    print(f"  {DIM}Image:{RESET}   {BOLD}{image}{RESET}")
    print(f"  {DIM}Output:{RESET}  {run_dir}")
    print(f"{CYAN}{'─' * 60}{RESET}\n")

    task_queue: asyncio.Queue = asyncio.Queue()
    for t in tasks:
        await task_queue.put(t)

    results: list[dict] = list(prior_results) if prior_results else []
    results_lock = asyncio.Lock()

    async def launch_worker(i: int):
        # No host-side stagger: unlike the local runner (which spreads Chromium
        # startups on one host), here the browser runs in the remote container,
        # so all workers acquire concurrently and the cluster's admission +
        # capacity govern real concurrency. A small stagger only avoids a
        # thundering herd of acquire RPCs.
        if args.worker_stagger:
            await asyncio.sleep(i * args.worker_stagger)
        await container_worker(
            i,
            task_queue,
            results,
            results_lock,
            client=client,
            image=image,
            args=args,
            run_dir=run_dir,
            web_app_rel=web_app_rel,
            web_app_dir=web_app_dir,
            env_vars=env_vars,
        )

    if num_workers:
        await asyncio.gather(*(launch_worker(i) for i in range(num_workers)))

    if not results:
        return [], {}

    aggregate, report_path = _aggregate_and_report(
        results, args, run_dir, timestamp, num_workers
    )

    pct = aggregate["pass_rate"]
    pct_color = GREEN if pct >= 50 else RED
    by_diff = aggregate["by_difficulty"]
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"  {BOLD}Results{run_label}: {pct_color}{aggregate['passed']}/{aggregate['total']} passed ({pct}%){RESET}")
    print()
    for d in ["easy", "medium", "hard"]:
        if d in by_diff:
            info = by_diff[d]
            dc = DIFF_COLOR.get(d, "")
            rc = GREEN if info["passed"] == info["total"] else (YELLOW if info["passed"] > 0 else RED)
            print(f"    {dc}{d.capitalize():8s}{RESET} {rc}{info['passed']}/{info['total']}{RESET}")
    print()
    print(f"  {DIM}Report:{RESET} {MAGENTA}{report_path}{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}\n")

    return results, aggregate


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(description="xrlenv parallel evaluation runner")
    # Shared with run_eval_parallel.py. 'oracle' is a browserless pseudo-agent
    # that PUTs the solved state — runs the identical pipeline (server reuse +
    # reset) with no LLM, for cheap infra + verifier validation.
    parser.add_argument("--model", choices=list(AGENT_FACTORIES.keys()) + ["oracle"], default="gpt")
    parser.add_argument("--task-id", default=None,
                        help="Run one or more tasks, comma-separated")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--use-vision", action="store_true")
    add_monet_args(parser, include_bin=False)  # cluster clones Monet to a fixed path
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel containers (capped at the task count)")
    parser.add_argument("--worker-stagger", type=float, default=0.0,
                        help="Seconds between worker startups (default 0 = all acquire at once)")
    parser.add_argument("--output-dir", default=None,
                        help="Results directory (default: <web-app>/results) — matches "
                             "run_eval_parallel.py so the pipeline/collection find it")
    parser.add_argument("--web-app", default="apps/gitlab-org-management")
    parser.add_argument("--task-suite", default="real-tasks")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--failed-only", action="store_true")
    parser.add_argument("--no-skip-timeout", action="store_true")
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--tag", default=None)
    # xrlenv-specific. Host/port default to None and fall back to .env (via
    # xrlenv_config) — pass them only to override the .env coordinates.
    parser.add_argument("--image", default=None,
                        help="Full image ref (default: from .env registry)")
    parser.add_argument("--xrlenv-host", default=None,
                        help="Control-plane gRPC host override (default: .env)")
    parser.add_argument("--xrlenv-port", type=int, default=None,
                        help="Control-plane gRPC port override (default: .env)")
    parser.add_argument("--acquire-timeout", type=float, default=1800.0,
                        help="Per-container acquire timeout (s); raise for cold image pulls")
    parser.add_argument("--env-file", default=None,
                        help="Extra .env of LLM API keys to forward (in addition to project .env)")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]),
                        help="Repo root on host (for computing apps/<app> path)")
    parser.add_argument("--eval-dir", default=str(Path(__file__).resolve().parent),
                        help="Local evaluation/ dir injected into each container")
    args = parser.parse_args()
    # --monet-* flags override .env BEFORE monet_env() reads the environment, so
    # the MONET_* forwarded into the container reflect them. (Shared with the
    # local runner — see run_eval_parallel.apply_monet_overrides.)
    apply_monet_overrides(args)

    image = args.image or default_image()
    if not image:
        print(f"{RED}No --image and no registry in .env (XRLENV_PRIVATE_REGISTRY_HOST).{RESET}")
        sys.exit(1)

    web_app_dir = str(Path(args.web_app).resolve())
    # Default + --output-dir match run_eval_parallel.py exactly, so the pipeline
    # (find_latest_results, parse_results, S3 upload) and collect_results work
    # unchanged. Use --output-dir to redirect a manual sweep elsewhere.
    output_dir = args.output_dir or os.path.join(web_app_dir, "results")
    env_vars = llm_env(args.env_file)
    if args.model == "monet":
        missing = monet_preflight()
        if missing:
            print(f"{RED}--model monet prerequisites missing (set them in .env):{RESET}")
            for m in missing:
                print(f"  - {m}")
            print(f"{DIM}  See docs/RUNBOOK-monet.md (§5 xrlenv cluster mode).{RESET}")
            sys.exit(1)
        env_vars.update(monet_env())  # gateway creds + MONET_* + GH_TOKEN + MONET_REPO

    all_tasks = load_tasks(web_app_dir, task_suite=args.task_suite)
    tasks = filter_tasks(all_tasks, task_id=args.task_id, difficulty=args.difficulty)
    if not tasks:
        print(f"{RED}No tasks matched.{RESET}")
        sys.exit(1)

    try:
        client = make_client(args.xrlenv_host, args.xrlenv_port)
    except RuntimeError as e:
        print(f"{RED}{e}{RESET}")
        sys.exit(1)

    try:
        if args.repetitions <= 1:
            await _run_once(args, tasks, output_dir, web_app_dir, client, image, env_vars)
        else:
            await _run_multi(args, tasks, output_dir, web_app_dir, client, image, env_vars)
    finally:
        await client.close()


async def _run_once(args, tasks, output_dir, web_app_dir, client, image, env_vars):
    if args.resume_dir:
        run_dir = Path(args.resume_dir)
        remaining, prior = find_incomplete_tasks(run_dir, tasks)
        if not remaining:
            print(f"{GREEN}All tasks already completed in {run_dir}{RESET}")
            return
        results, _ = await run_single_eval(
            remaining, args, run_dir, web_app_dir, client, image, env_vars, prior_results=prior
        )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suite_tag = f"_{args.task_suite}" if args.task_suite != "real-tasks" else ""
        extra_tag = f"_{args.tag}" if args.tag else ""
        run_dir = Path(output_dir) / f"{args.model}_{timestamp}{suite_tag}{extra_tag}_xrlenv"
        run_dir.mkdir(parents=True, exist_ok=True)
        results, _ = await run_single_eval(
            tasks, args, run_dir, web_app_dir, client, image, env_vars
        )
    if not results:
        print(f"{RED}No tasks completed.{RESET}")
        sys.exit(1)


async def _run_multi(args, tasks, output_dir, web_app_dir, client, image, env_vars):
    num_workers = min(args.workers, len(tasks))
    if args.resume_dir:
        parent_dir = Path(args.resume_dir)
        parts = parent_dir.name.split("_", 1)
        timestamp = parts[1] if len(parts) > 1 else datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suite_tag = f"_{args.task_suite}" if args.task_suite != "real-tasks" else ""
        extra_tag = f"_{args.tag}" if args.tag else ""
        parent_dir = Path(output_dir) / f"{args.model}_{timestamp}{suite_tag}{extra_tag}_xrlenv"
    parent_dir.mkdir(parents=True, exist_ok=True)

    mode_label = "cascading (failed-only)" if args.failed_only else "full repeat"
    print(f"\n{BOLD}{MAGENTA}{'=' * 60}{RESET}")
    print(f"{BOLD}{MAGENTA}  Multi-Run xrlenv Eval: {args.repetitions} rounds ({mode_label}){RESET}")
    print(f"  {DIM}Output:{RESET}  {parent_dir}")
    print(f"{BOLD}{MAGENTA}{'=' * 60}{RESET}")

    run_dirs: list[Path] = []
    current_tasks = tasks
    all_tasks_by_id = {t["id"]: t for t in tasks}

    for rep in range(1, args.repetitions + 1):
        if args.failed_only and not current_tasks:
            print(f"\n{GREEN}All tasks passed — stopping early after {rep - 1} runs.{RESET}")
            break
        run_dir = parent_dir / f"run{rep}"
        run_dirs.append(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        results, _ = await run_single_eval(
            current_tasks, args, run_dir, web_app_dir, client, image, env_vars,
            run_label=f" (run {rep}/{args.repetitions}, {len(current_tasks)} tasks)",
        )
        if not results:
            print(f"{YELLOW}Run {rep}: no tasks completed, skipping.{RESET}")
            continue
        if args.failed_only:
            skip_timeout = not args.no_skip_timeout
            failed_ids = {
                r["task_id"] for r in results
                if not r["passed"] and not (skip_timeout and not r.get("is_done", True))
            }
            current_tasks = [all_tasks_by_id[t] for t in failed_ids if t in all_tasks_by_id]

    if not run_dirs:
        print(f"{RED}No runs completed.{RESET}")
        sys.exit(1)

    merged_dir = parent_dir / "merged"
    print(f"\n{BOLD}{MAGENTA}  Merging {len(run_dirs)} runs → merged/{RESET}")
    aggregate, _ = merge_repetition_results(
        run_dirs=run_dirs,
        merged_dir=merged_dir,
        model=args.model,
        timestamp=timestamp,
        use_vision=args.use_vision,
        num_workers=num_workers,
    )
    shutil.copy2(merged_dir / "results.json", parent_dir / "results.json")
    shutil.copy2(merged_dir / "report.html", parent_dir / "report.html")

    pct = aggregate["pass_rate"]
    pct_color = GREEN if pct >= 50 else RED
    print(f"\n{BOLD}{MAGENTA}{'=' * 60}{RESET}")
    print(f"  {BOLD}Merged Results ({len(run_dirs)} runs): "
          f"{pct_color}{aggregate['passed']}/{aggregate['total']} passed ({pct}%){RESET}")
    print(f"  {DIM}Output:{RESET}   {parent_dir}")
    print(f"{MAGENTA}{'=' * 60}{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
