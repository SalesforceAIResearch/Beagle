#!/usr/bin/env python3
"""Parallel oracle / real-agent sweep across EvoClaw repos — the batch driver.

The whole point of xrlenv is concurrency, so this runs tasks CONCURRENTLY with a
controllable cap. Foreground (NOT detached): a bounded ``ThreadPoolExecutor``
launches one ``run_e2e_xrlenv.py`` subprocess per task, streams live start/done
progress, then prints a summary table. (Port of the former ``run_oracle_sweep.sh``.)

Two parallelization levels (--parallelization-level):
  repo       (default) task = a whole repo (its full selected milestone set).
             Concurrency = repos in flight; each repo internally evaluates up to 4
             milestones in parallel (EvoClaw's fixed ThreadPoolExecutor). Agent
             ContainerSetup paid once per repo.
  milestone  task = ONE milestone. Flatten X repos -> Y milestone tasks, pool of N.
             Each task builds a single-milestone workspace (unique trial_root/lock)
             and runs it faithfully (own agent + eval container). Exact N-way
             control, no long tail; pays one ContainerSetup per milestone.

Concurrency: --workers N (default = #tasks for --parallelization-level repo, else
8). Each task is an independent process (own shim, own xrlenv client, own
``xrl-<pid>-`` name prefix). The CP's AIMD admission queues any excess (never
fails). Repos are chosen with --repos (default: every dataset repo); arbitrary
run_e2e wrapper flags are forwarded verbatim after a ``--`` separator.

--run-name NAME is REQUIRED: each run gets its own results dir
<results-root>/<NAME>__<UTC-timestamp>/ so runs are separated + comparable. The
shared-dataset symlinks (task metadata) are STRIPPED from each run's results; only
EvoClaw's real outputs (e2e_trial: eval logs, agent trajectory, verdicts) are kept,
for a clean per-run audit trail.

Usage (from the EvoClaw checkout root, with .env_private configured):
    python xrlenv_onboard/run_all_xrlenv.py --run-name smoke                                    # all repos, per-repo
    python xrlenv_onboard/run_all_xrlenv.py --run-name smoke --workers 4
    python xrlenv_onboard/run_all_xrlenv.py --parallelization-level milestone --run-name all     # all milestones, pool of 8
    python xrlenv_onboard/run_all_xrlenv.py --parallelization-level milestone --run-name sub --repos navidrome_... zeromicro_...
    python xrlenv_onboard/run_all_xrlenv.py --parallelization-level milestone --workers 64 --run-name r1 -- --keep-container  # worker flag via --
    nohup python xrlenv_onboard/run_all_xrlenv.py --parallelization-level milestone --workers 32 --run-name overnight >sweep.out 2>&1 &

Fleet reservation is ON by default (--no-fleet to disable): it reserves each task's
whole container-fleet footprint (agent + evals) on one node so the eval containers
aren't starved by greedy admission of the many agent containers. The footprint is
AUTO-derived from EvoClaw's spec + the parallelization level's fan-out (milestone ->
cpu 18, repo -> cpu 66) — you type NO numbers — and PRINTED prominently at launch
(never a silent default). Preview it with --dry-run; override any input with
--fleet-eval-cpu / --fleet-agent-cpu / --fleet-eval-pool / --mem-per-cpu-gb,
or pin the total with --fleet-footprint-cpu/-mem-gb.
    python xrlenv_onboard/run_all_xrlenv.py --parallelization-level milestone --workers 64 --run-name r2
    python xrlenv_onboard/run_all_xrlenv.py --parallelization-level milestone --run-name r3 --dry-run  # preview footprint, launch nothing

Needs EVOCLAW_DATA_ROOT + a results root (--results-root, default <project>/results)
+ cluster vars in .env / .env_private.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# The EvoClaw checkout root (parent of xrlenv_onboard/) — same robust derivation
# scripts/run_all.py uses (Path(__file__).resolve().parent.parent).
_PROJECT_ROOT = _HERE.parent

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))  # env_loader / workspace siblings


def _quarantine_env(repo: str) -> dict[str, str]:
    """Per-repo anti-cheat ("quarantine") env, FAITHFUL to upstream
    ``scripts/run_all.py`` (which does ``worker_env = {**os.environ, **q_env}``).

    Upstream's harness is now fail-closed: if ``quarantine_configs/<repo>.yaml``
    exists but the worker has no ``EVOCLAW_QUARANTINE`` env, ``run_e2e`` refuses
    to run (scores would be tainted). The policy also forces the eval offline
    (``GOPROXY=off`` + deny the repo's own registry domains), which is what makes
    scoring deterministic. We only take over the *docker* job, so we mirror
    upstream verbatim: compute the repo's policy env and merge it into the WORKER
    subprocess env; the unmodified harness applies the isolation inside the
    container. Returns ``{}`` when the repo has no quarantine config (off)."""
    try:
        from harness.e2e.quarantine import (  # type: ignore[import-not-found]
            load_quarantine_env,
        )
        return load_quarantine_env(repo, _PROJECT_ROOT)
    except SystemExit as exc:  # malformed policy → upstream sys.exits; isolate to this task
        print(f"[run_all] WARN: quarantine policy for {repo} is malformed ({exc}); "
              "this task will hit the harness fail-closed gate.")
        return {}


def _default_results_root() -> str:
    """Default per-run results root: ``<project-root>/results``.
    Overridden by ``--results-root``."""
    return str(_PROJECT_ROOT / "results")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Consume the sweep's own flags. Positional args are repo-name filters; a
    ``--`` separator forwards everything after it verbatim to each worker."""
    # Split off the passthru at the FIRST bare `--` ourselves (argparse.REMAINDER
    # interacts badly with an optional positional — it swallows the `--` into the
    # positional). Everything after `--` is forwarded verbatim to each worker; the
    # rest is parsed normally. This mirrors the sweep's explicit `--) PASSTHRU`.
    passthru: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, passthru = argv[:cut], argv[cut + 1:]
    # allow_abbrev=False so a positional repo filter can never be prefix-matched
    # onto one of our flags. --run-name is required (fail loud) — enforced below.
    p = argparse.ArgumentParser(
        prog="run_all_xrlenv.py", add_help=True, allow_abbrev=False,
        description="Parallel oracle / real-agent sweep across EvoClaw repos (batch driver).",
    )
    p.add_argument("--parallelization-level", dest="by", choices=("repo", "milestone"),
                   default="repo",
                   help="task granularity: 'repo' (one task per repo, coarser) or "
                        "'milestone' (one task per milestone, finer — saturates a big cluster)")
    p.add_argument("--agent", default="oracle", metavar="AGENT",
                   help="which agent every worker runs (default: 'oracle' — the no-LLM "
                        "golden-solution oracle). Pass a real agent (claude-code / codex / "
                        "gemini-cli / openhands) with --model + UNIFIED_API_KEY/_BASE_URL to "
                        "sweep it through the SAME batch scaffolding (fleet reservation, "
                        "per-repo quarantine, cpu-pinning, --apply-yd-fixes, mem caps, results "
                        "layout). Agent-specific flags go via a trailing '-- ...'.")
    p.add_argument("--model", default="none", metavar="MODEL",
                   help="model for --agent (default: 'none', which the oracle wants). "
                        "REQUIRED (a real value, not 'none') when --agent is not 'oracle'.")
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="number of tasks run concurrently "
                        "(default: #tasks for --parallelization-level repo, else 8)")
    p.add_argument("--run-name", default=None, metavar="NAME",
                   help="REQUIRED: names this run; results land in "
                        "<results-root>/<NAME>__<UTC-timestamp>/ (per-run, comparable)")
    p.add_argument("--results-root", default=_default_results_root(), metavar="DIR",
                   help="per-run results parent dir (default: <project-root>/results)")
    p.add_argument("--copy-testbed", action="store_true",
                   help="forward EvoClaw's whole-/testbed debug copy to every task (default: OFF)")
    p.add_argument("--apply-yd-fixes", action="store_true",
                   help="forward --apply-yd-fixes to every worker: opt-in local "
                        "corrections to known UPSTREAM eval-protocol bugs (preserve "
                        "untracked GT test files across the evaluator's git clean, e.g. "
                        "element e662c19/fba5938). Default OFF = faithful, "
                        "leaderboard-comparable. See xrlenv_onboard/yd_fixes.py.")
    p.add_argument("--cpu-pinning-milestone", action="append", default=None,
                   dest="cpu_pinning_milestones", metavar="[REPO/]MID",
                   help="enable xrlenv cpuset-pinning for THIS milestone's worker only "
                        "(each of its containers gets ceil(cpus) dedicated cores, so "
                        "nproc==cpus) — stops go/cargo/jest oversubscribing on big nodes "
                        "(the Table A contention set). Repeatable; matches by mid, "
                        "repo/mid, or repo__mid — symmetric to --sysbox-milestone. "
                        "Everything else keeps the CFS-quota default. Onboarding "
                        "runtime-patch, no xrlenv-core change. See cpu_pinning.py.")
    # Fleet reservation (default ON — it is the reason this driver exists). AUTO-
    # derives each task's peak footprint from EvoClaw's spec + the parallelization
    # level's fan-out; every input is overridable so a change to EvoClaw's sizing is
    # a one-flag fix — and the derived footprint is PRINTED at launch, never silent.
    p.add_argument("--fleet", action=argparse.BooleanOptionalAction, default=True,
                   help="reserve each task's whole container fleet (agent + evals) on one "
                        "node so evals aren't starved by greedy admission "
                        "(default: ON; --no-fleet for legacy per-container admission)")
    p.add_argument("--fleet-footprint-cpu", type=float, default=None, metavar="C",
                   help="explicit override: pin the task's peak cpu footprint "
                        "(with --fleet-footprint-mem-gb; else auto-derived)")
    p.add_argument("--fleet-footprint-mem-gb", type=float, default=None, metavar="G",
                   help="explicit override: pin the task's peak memory footprint (GiB)")
    # EvoClaw's own container-fleet spec — the source of truth for AUTO footprint
    # derivation. A task = 1 long-lived agent container + N eval containers (N =
    # fan-out: 1 per-milestone, POOL per-repo). These are EvoClaw facts (the agent
    # admits at xrlenv's default cpu; the eval runs `--cpus 16`; per-repo evals run
    # in a fixed ThreadPoolExecutor). All overridable so a change to EvoClaw's
    # sizing is a one-flag fix.
    p.add_argument("--fleet-agent-cpu", type=float, default=2.0, metavar="C",
                   help="agent container cpu (xrlenv default cpu_request, no --cpus; default: 2)")
    p.add_argument("--fleet-eval-cpu", type=float, default=16.0, metavar="C",
                   help="eval container cpu (EvoClaw's `--cpus 16`; default: 16)")
    p.add_argument("--fleet-eval-pool", type=int, default=4, metavar="N",
                   help="per-repo fan-out (EvoClaw's fixed ThreadPoolExecutor(4); default: 4)")
    p.add_argument("--mem-per-cpu-gb", type=float, default=2.0, metavar="G",
                   help="memory per CPU, GiB — sets BOTH each container's --memory cap and "
                        "the fleet reservation memory (footprint cpu x this); default: 2")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan (incl. derived footprint) and exit, launch nothing")
    p.add_argument("--repos", nargs="+", default=None, dest="filters", metavar="REPO",
                   help="restrict the run to these repo names (default: every dataset repo)")
    p.add_argument("--milestones", nargs="+", default=None, dest="milestone_filters",
                   metavar="MID",
                   help="restrict a --parallelization-level milestone run to these "
                        "milestone ids (match by mid or repo/mid). Symmetric to "
                        "--repos; for targeted experiment runs.")
    p.add_argument("--sysbox-milestone", action="append", default=None,
                   dest="sysbox_milestones", metavar="[REPO/]MID",
                   help="route THIS milestone to the Sysbox (sysbox-runc) node pool "
                        "for unprivileged Docker-in-Docker — e.g. element-web's "
                        "testcontainers E2E milestone. Repeatable; matches by mid, "
                        "repo/mid, or repo__mid. Every other task stays on the "
                        "default runc pool (so DinD tasks don't monopolise the small "
                        "sysbox pool). Best with --parallelization-level milestone. "
                        "Requires an operator Sysbox pool (xrlenv_plugins/sysbox/) + "
                        "sysbox-runc in nodes.yaml policy.allowed_runtimes.")
    args = p.parse_args(argv)
    args.passthru = passthru

    # --run-name is required — fail fast, before anything else runs.
    if not args.run_name:
        p.error(
            "--run-name NAME is required — give this run a name; results go to "
            "<results-root>/<name>__<UTC-timestamp>/ (per-run, comparable)."
        )
    # A real agent needs a real model — the 'none' default is oracle-only.
    if args.agent != "oracle" and (not args.model or args.model == "none"):
        p.error(
            f"--agent {args.agent} needs a real --model (got '{args.model}'); "
            "'none' is only for --agent oracle. Also set UNIFIED_API_KEY / "
            "UNIFIED_BASE_URL for a real agent."
        )
    return args


def _fmt(x: float) -> str:
    """Match awk's %g: drop a trailing ``.0`` so 18.0 prints as 18 (like the sweep)."""
    return f"{x:g}"


def _resolve_fleet_footprint(args: argparse.Namespace) -> tuple[float | None, float | None, str, int]:
    """Derive the per-task fleet footprint (or take the explicit override).

    Returns ``(cpu, mem_gb, source_label, fanout)``. When ``--fleet`` is off,
    ``cpu`` / ``mem_gb`` are ``None`` and this prints nothing. Mirrors the sweep's
    both-or-neither rule + the big "FLEET RESERVATION: ON" box.
    """
    if not args.fleet:
        return None, None, "", 0

    # fan-out per task: per-milestone runs 1 eval; per-repo runs up to POOL.
    fanout = 1 if args.by == "milestone" else args.fleet_eval_pool
    cpu, mem_gb = args.fleet_footprint_cpu, args.fleet_footprint_mem_gb
    if cpu is not None and mem_gb is not None:
        source = "explicit override (--fleet-footprint-cpu/-mem-gb)"
    elif cpu is not None or mem_gb is not None:
        raise SystemExit(
            "--fleet-footprint-cpu and --fleet-footprint-mem-gb must be given together\n"
            "  (or neither — then the footprint auto-derives from EvoClaw's spec)."
        )
    else:
        # AUTO: footprint = agent + fan-out x eval; mem = footprint_cpu x mem-per-cpu
        # (the same cpuxmem model xrlenv applies per container).
        cpu = args.fleet_agent_cpu + fanout * args.fleet_eval_cpu
        mem_gb = cpu * args.mem_per_cpu_gb
        source = (f"auto: agent {_fmt(args.fleet_agent_cpu)} + {fanout}xeval "
                  f"{_fmt(args.fleet_eval_cpu)} cpu; mem = cpu x {_fmt(args.mem_per_cpu_gb)}")

    bar = "═" * 70
    print(bar)
    print(f"  FLEET RESERVATION: ON   (--parallelization-level {args.by}, fan-out {fanout} eval container(s)/task)")
    print(f"  footprint reserved per task:  cpu={_fmt(cpu)}  mem={_fmt(mem_gb)} GiB")
    print(f"  derivation: {source}")
    print("  (each task reserves this whole footprint on ONE node so its eval")
    print("   container(s) can't be starved by other tasks' agents. Override with")
    print("   --fleet-eval-cpu / --fleet-agent-cpu / --fleet-eval-pool /")
    print("   --mem-per-cpu-gb, or pin --fleet-footprint-cpu/-mem-gb.)")
    print(bar)
    return cpu, mem_gb, source, fanout


def _discover_repos(data_root: Path, filters: list[str]) -> list[str]:
    """Repo list: the filter args, else every dataset dir with a metadata.json."""
    if filters:
        return list(filters)
    repos = [
        p.parent.name
        for p in sorted(data_root.glob("*/metadata.json"))
        if p.is_file()
    ]
    return sorted(repos)


def _build_tasks(
    by: str, repos: list[str], data_root: Path,
    milestones: list[str] | None = None,
) -> list[tuple[str, str | None]]:
    """Task list: ``(repo, None)`` per-repo, or ``(repo, mid)`` per-milestone.

    Per-milestone flattens each repo's ``selected_milestone_ids.txt`` (skipping
    blank / ``#`` lines) into one task per id. ``milestones`` (from
    ``--milestones``) restricts to those ids (matched by ``mid`` or
    ``repo/mid``) — symmetric to ``--repos``, for targeted experiment runs.
    """
    if by == "repo":
        return [(repo, None) for repo in repos]

    def _ids(path: Path) -> list[str]:
        if not path.is_file():
            return []
        out = []
        for raw in path.read_text().splitlines():
            mid = "".join(raw.split())  # strip ALL whitespace (matches the sweep)
            if mid and not mid.startswith("#"):
                out.append(mid)
        return out

    mfilter = set(milestones or ())
    tasks: list[tuple[str, str | None]] = []
    for repo in repos:
        sel = data_root / repo / "selected_milestone_ids.txt"
        if not sel.is_file():
            print(f"warn: no selected_milestone_ids.txt for {repo} — skipping", file=sys.stderr)
            continue
        # GRADED set = selected - non-graded. The EvoClaw paper grades 98 milestones;
        # `selected_milestone_ids.txt` is graded+context, and each repo's
        # `non-graded_milestone_ids.txt` marks the context milestones to EXCLUDE from
        # scoring (3 total: dubbo 1, ripgrep 2). Running/scoring those inflates the
        # denominator and doesn't match the paper — so we drop them here.
        nongraded = set(_ids(data_root / repo / "non-graded_milestone_ids.txt"))
        graded = [m for m in _ids(sel) if m not in nongraded]
        dropped = len(nongraded & set(_ids(sel)))
        if dropped:
            print(f"  {repo}: excluded {dropped} non-graded context milestone(s) "
                  f"(graded={len(graded)})", file=sys.stderr)
        for mid in graded:
            if mfilter and mid not in mfilter and f"{repo}/{mid}" not in mfilter:
                continue
            tasks.append((repo, mid))
    return tasks


def _parse_completed_failed(log_text: str) -> tuple[str, str]:
    """Parse the last ``Completed: N`` / ``Failed: N`` off a task log (like the sweep)."""
    def _last(pat: str) -> str:
        m = re.findall(pat, log_text)
        return m[-1] if m else "?"
    return _last(r"Completed: (\d+)"), _last(r"Failed: (\d+)")


def _strip_dataset_symlinks(ws_task: Path, data_root: Path) -> None:
    """Strip the shared-dataset symlinks (pointers into EVOCLAW_DATA_ROOT — the task
    metadata: metadata.json/dependencies.csv/dockerfiles/milestones.csv/...) from
    this run's results, keeping ONLY EvoClaw's real outputs (e2e_trial: eval logs,
    agent trajectory, verdicts) for a clean per-run audit trail. unlink() removes
    the symlink, never the shared data it points at. Equivalent to the sweep's
    ``find -type l -lname "$DATA_ROOT/*" -delete``.
    """
    if not ws_task.is_dir():
        return
    data_root_str = str(data_root)
    for root, dirs, files in os.walk(ws_task):
        # os.walk does not descend into symlinked dirs (followlinks=False), so a
        # symlinked `dockerfiles/` is unlinked here as an entry, never walked into.
        for name in (*dirs, *files):
            path = Path(root) / name
            if not path.is_symlink():
                continue
            try:
                target = os.readlink(path)
            except OSError:
                continue
            if target.startswith(data_root_str + os.sep):
                with contextlib.suppress(OSError):
                    path.unlink()


class _Task:
    """One sweep task: builds its workspace, runs one run_e2e_xrlenv subprocess,
    tees output to a log, parses the result, and strips the dataset symlinks."""

    def __init__(
        self,
        repo: str,
        mid: str | None,
        *,
        data_root: Path,
        run_ws: Path,
        log_dir: Path,
        fleet_cpu: float | None,
        fleet_mem_gb: float | None,
        mem_per_cpu_gb: float,
        copy_testbed: bool,
        passthru: list[str],
        agent: str = "oracle",
        model: str = "none",
        sysbox_milestones: frozenset[str] = frozenset(),
        cpu_pinning_milestones: frozenset[str] = frozenset(),
    ) -> None:
        self.repo = repo
        self.mid = mid
        self.agent = agent
        self.model = model
        self.label = f"{repo}__{mid}" if mid else repo
        # Per-milestone Sysbox routing (opt-in). A task runs under sysbox-runc
        # (unprivileged Docker-in-Docker) ONLY when its milestone is in the
        # allowlist — matched forgivingly by mid, repo/mid, or repo__mid. This
        # keeps DinD tasks on the small sysbox pool while everything else uses
        # the full runc pool (no compute waste). Per-repo tasks (mid is None)
        # bundle many milestones in one worker, so runtime can't be scoped —
        # they never auto-enable sysbox.
        self.sysbox = bool(mid) and (
            mid in sysbox_milestones
            or f"{repo}/{mid}" in sysbox_milestones
            or f"{repo}__{mid}" in sysbox_milestones
        )
        # Per-milestone cpuset-pinning (opt-in), same forgiving match as sysbox.
        # Scopes pinning to the flagged milestone's worker so ONLY the contention-
        # sensitive evals pin dedicated cores; every other task keeps the CFS-quota
        # default. Per-repo tasks (mid None) bundle many milestones → never scoped.
        self.cpu_pinning = bool(mid) and (
            mid in cpu_pinning_milestones
            or f"{repo}/{mid}" in cpu_pinning_milestones
            or f"{repo}__{mid}" in cpu_pinning_milestones
        )
        self.data_root = data_root
        self.run_ws = run_ws
        self.log = log_dir / f"{self.label}.log"
        self.result_file = log_dir / f"{self.label}.result"
        self.fleet_cpu = fleet_cpu
        self.fleet_mem_gb = fleet_mem_gb
        self.mem_per_cpu_gb = mem_per_cpu_gb
        self.copy_testbed = copy_testbed
        self.passthru = passthru
        # filled in by run()
        self.rc = 1
        self.seconds = 0
        self.completed = "?"
        self.failed = "?"

    def _worker_cmd(self, ws: Path) -> list[str]:
        cmd = [
            sys.executable, str(_HERE / "run_e2e_xrlenv.py"),
            "--agent", self.agent, "--model", self.model, "--force",
            "--repo-name", self.repo, "--workspace-root", str(ws),
            # one rate drives both the fleet reservation (above) and the worker's
            # per-container --memory cap, so they can never drift.
            "--mem-per-cpu-gb", _fmt(self.mem_per_cpu_gb),
        ]
        if self.fleet_cpu is not None and self.fleet_mem_gb is not None:
            cmd += ["--fleet-footprint-cpu", _fmt(self.fleet_cpu),
                    "--fleet-footprint-mem-gb", _fmt(self.fleet_mem_gb)]
        if self.copy_testbed:
            cmd.append("--copy-testbed")
        if self.cpu_pinning:
            cmd.append("--cpu-pinning")
        cmd += self.passthru
        return cmd

    def run(self) -> _Task:
        import workspace  # sibling — import link_workspace directly (no subprocess)

        print(f"[{time.strftime('%H:%M:%S')}] start {self.label}")
        t0 = time.time()
        # Build this task's workspace. Per-milestone: base = <run-ws>/<repo>__<mid>,
        # single_milestone=mid (its own trial_root/lock, one-line
        # selected_milestone_ids.txt). Per-repo: base = <run-ws>.
        if self.mid:
            base = self.run_ws / f"{self.repo}__{self.mid}"
            ws_task = base
        else:
            base = self.run_ws
            ws_task = self.run_ws / self.repo
        with open(self.log, "w") as logf:
            try:
                ws = workspace.link_workspace(
                    self.data_root, base, self.repo, single_milestone=self.mid,
                )
            except SystemExit as exc:
                logf.write(f"FAIL build workspace: {exc}\n")
                self.rc = 3
                ws = None
            if ws is not None:
                logf.flush()
                # Per-task env: the driver is authoritative for the container
                # runtime. Set sysbox-runc only for allowlisted DinD milestones;
                # explicitly UNSET it otherwise, so a stray global export can't
                # route a non-DinD task onto the small sysbox pool.
                env = dict(os.environ)
                # Per-repo quarantine (anti-cheat) — mirror upstream
                # scripts/run_all.py: merge this repo's policy env into the worker
                # so the harness applies offline isolation inside the container.
                env.update(_quarantine_env(self.repo))
                if self.sysbox:
                    env["EVOCLAW_CONTAINER_RUNTIME"] = "sysbox-runc"
                    logf.write("[run_all] routing to sysbox-runc pool (DinD)\n")
                    logf.flush()
                else:
                    env.pop("EVOCLAW_CONTAINER_RUNTIME", None)
                proc = subprocess.run(
                    self._worker_cmd(ws), cwd=str(_PROJECT_ROOT),
                    stdout=logf, stderr=subprocess.STDOUT, env=env,
                )
                self.rc = proc.returncode
        # Strip the shared-dataset symlinks from this task's workspace subtree,
        # keeping e2e_trial + all real outputs.
        _strip_dataset_symlinks(ws_task, self.data_root)
        self.seconds = int(time.time() - t0)
        log_text = self.log.read_text(errors="replace") if self.log.is_file() else ""
        self.completed, self.failed = _parse_completed_failed(log_text)
        self.result_file.write_text(
            f"{self.label}\t{self.seconds}s\t{self.completed}\t{self.failed}\t{self.rc}\n"
        )
        print(f"[{time.strftime('%H:%M:%S')}] done  {self.label} "
              f"({self.seconds}s, completed={self.completed} failed={self.failed} rc={self.rc})")
        return self


def _write_xrlenv_summary(run_ws: Path, level: str) -> Path:
    """Write ONE eyeball-friendly nested JSON of the run's *real* result — milestones
    RESOLVED, not just rc=0 — to ``<run>/xrlenv_summary.json`` (the ``xrlenv_`` prefix
    marks it as added by us, not EvoClaw). Built by walking each milestone's
    ``evaluation_result.json``; retries fold in (a milestone counts resolved if ANY
    attempt resolved). Repos are ordered worst-first and milestones unresolved-first,
    so problems surface at the top instead of being buried in the per-task tree.
    """
    # per-task wall-clock + rc, keyed by the sweep-log label (<repo>__<mid>).
    secs_rc: dict[str, tuple[int, int]] = {}
    for rf in (_HERE / "tmp" / f"sweep-logs-{level}").glob("*.result"):
        try:
            label, secs, _c, _f, rc = rf.read_text().strip().split("\t")
            secs_rc[label] = (int(secs.rstrip("s")), int(rc))
        except (ValueError, OSError):
            continue

    # per (repo, milestone_id): resolved (OR across retries) + best test counts.
    repos: dict[str, dict[str, dict]] = {}
    for root, dirs, files in os.walk(run_ws):
        dirs[:] = [d for d in dirs if d not in ("testbed", "e2e_workspace", ".git", "node_modules")]
        if "evaluation_result.json" not in files:
            continue
        erj = Path(root) / "evaluation_result.json"
        try:
            d = json.loads(erj.read_text())
            repo = erj.parts[erj.parts.index("e2e_trial") - 1]
        except (ValueError, json.JSONDecodeError, OSError):
            continue
        mid = str(d.get("milestone_id") or "?")
        ts = d.get("tests_status") or {}
        passed = sum(len(v.get("success", [])) for v in ts.values() if isinstance(v, dict))
        failed = sum(len(v.get("failure", [])) for v in ts.values() if isinstance(v, dict))
        cur = repos.setdefault(repo, {}).setdefault(mid, {"resolved": False, "passed": 0, "failed": 0})
        cur["resolved"] = cur["resolved"] or bool(d.get("resolved"))
        if passed + failed >= cur["passed"] + cur["failed"]:  # keep the real (non-empty) attempt
            cur["passed"], cur["failed"] = passed, failed

    total = resolved = 0
    blocks: list[tuple[float, str, dict]] = []
    for repo, mids in repos.items():
        r = sum(1 for v in mids.values() if v["resolved"])
        n = len(mids)
        total += n
        resolved += r
        ordered = {}
        for mid in sorted(mids, key=lambda m: (mids[m]["resolved"], m)):  # unresolved first
            v = mids[mid]
            s, rc = secs_rc.get(f"{repo}__{mid}", (None, None))
            ordered[mid] = {"resolved": v["resolved"], "passed": v["passed"],
                            "failed": v["failed"], "seconds": s, "rc": rc}
        blocks.append((r / n if n else 1.0, repo,
                       {"resolved": f"{r} / {n}", "milestones": ordered}))

    # Accounting integrity: `total` above only counts milestones that PRODUCED an
    # evaluation_result.json. A task can run to rc=0 yet yield NO verdict (eval
    # errored / crashed) — e.g. nushell milestone_G04_1ddae02. Silently that shrinks
    # the denominator (89 launched -> "75/88") and a real pipeline gap slips through.
    # So we reconcile LAUNCHED (one .result per task) against COLLECTED (has an
    # eval verdict) and fail loud on any shortfall. A launched label is covered if a
    # verdict matches it exactly (milestone level: <repo>__<mid>) or under it
    # (repo level: <repo>__*).
    launched = sorted(secs_rc)
    collected = {f"{repo}__{mid}" for repo, mids in repos.items() for mid in mids}
    no_verdict = [lbl for lbl in launched
                  if not any(ev == lbl or ev.startswith(lbl + "__") for ev in collected)]

    doc = {
        "run": run_ws.name,
        "parallelization_level": level,
        "totals": {
            "tasks_launched": len(launched),      # tasks the driver actually ran
            "milestones_evaluated": total,        # produced an eval verdict
            "no_verdict": len(no_verdict),        # ran but yielded NO verdict (a gap!)
            "resolved": resolved,
            "unresolved": total - resolved,
        },
        "no_verdict_tasks": no_verdict,           # surfaced, never buried
        "repos": {repo: block for _frac, repo, block in sorted(blocks, key=lambda x: (x[0], x[1]))},
    }
    out = run_ws / "xrlenv_summary.json"
    out.write_text(json.dumps(doc, indent=2))
    print(f"xrlenv_summary.json (RESOLVED {resolved}/{total} evaluated; "
          f"{len(launched)} launched): {out}")
    if no_verdict:
        print(f"  \033[0;31m⚠️  {len(no_verdict)} task(s) ran but produced NO eval "
              f"verdict (pipeline gap, NOT counted above):\033[0m {no_verdict}")
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # Load .env + .env_private (shell wins) so EVOCLAW_DATA_ROOT + the cluster vars
    # are available — same loader run_e2e_xrlenv uses.
    os.environ.setdefault("XRLENV_DOTENV", "off")
    import env_loader

    env_loader.load_project_dotenv()

    # The --apply-yd-fixes corrections repair KNOWN upstream eval-protocol bugs; without
    # them several milestones fail spuriously. Warn LOUD once at the top when they're off,
    # and mark the env the workers inherit so their run_e2e_xrlenv doesn't repeat the banner.
    if not args.apply_yd_fixes:
        import yd_fixes
        yd_fixes.warn_yd_fixes_off()
    os.environ["_XRLENV_YD_WARNED"] = "1"

    results_root = args.results_root
    data_root_str = os.environ.get("EVOCLAW_DATA_ROOT")
    if not data_root_str:
        raise SystemExit("set EVOCLAW_DATA_ROOT (see xrlenv_onboard/README.md)")
    data_root = Path(data_root_str).expanduser()

    # Fleet footprint (prints the big box when --fleet is on).
    fleet_cpu, fleet_mem_gb, _src, _fanout = _resolve_fleet_footprint(args)

    # Per-run results dir: <results-root>/<name>__<timestamp>/ (name required; the
    # timestamp guarantees no collision even with a repeated name).
    run_name_safe = re.sub(r"[^A-Za-z0-9._-]", "_", args.run_name)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_name_safe}__{stamp}"
    run_ws = Path(results_root).expanduser() / run_id  # this run's isolated results root

    repos = _discover_repos(data_root, args.filters)
    tasks = _build_tasks(args.by, repos, data_root, args.milestone_filters)
    if not tasks:
        print("no tasks")
        return 1

    workers = args.workers
    if workers is None:
        workers = len(tasks) if args.by == "repo" else 8

    if args.dry_run:
        # Print the plan + run-id + footprint and exit WITHOUT launching anything
        # AND without creating the run dir (the sweep's dry-run mkdir'd it — bug).
        print(f"DRY RUN — nothing launched: parallelization-level={args.by}, {len(tasks)} task(s), workers={workers}")
        print(f"  run-id: {run_id}  ->  results: {run_ws}")
        if fleet_cpu is not None and fleet_mem_gb is not None:
            print(f"  (fleet footprint per task: cpu={_fmt(fleet_cpu)} "
                  f"mem={_fmt(fleet_mem_gb)} GiB — see the FLEET box above)")
        return 0

    run_ws.mkdir(parents=True, exist_ok=True)
    log_dir = _HERE / "tmp" / f"sweep-logs-{args.by}"
    log_dir.mkdir(parents=True, exist_ok=True)
    for stale in log_dir.glob("*.result"):  # don't accumulate across re-runs
        stale.unlink()

    print(f"sweep [agent={args.agent} model={args.model}]: parallelization-level={args.by}, "
          f"{len(tasks)} task(s), workers={workers} -> logs in {log_dir}")
    print(f"  run-id: {run_id}")
    print(f"  results (per-run, dataset-symlinks stripped): {run_ws}")
    if args.by == "repo":
        print("  (each repo also evaluates up to 4 milestones in parallel internally)")

    _sysbox_set = frozenset(args.sysbox_milestones or ())
    _cpu_pin_set = frozenset(args.cpu_pinning_milestones or ())
    # Forward the opt-in global onboarding flags to every worker (prepend so they
    # survive any user-supplied ``-- ...`` passthrough that follows). --cpu-pinning is
    # NOT here — it's routed per-milestone via _Task (like --sysbox-milestone).
    passthru = (["--apply-yd-fixes"] if args.apply_yd_fixes else []) + list(args.passthru or [])
    task_objs = [
        _Task(
            repo, mid,
            data_root=data_root, run_ws=run_ws, log_dir=log_dir,
            fleet_cpu=fleet_cpu, fleet_mem_gb=fleet_mem_gb,
            mem_per_cpu_gb=args.mem_per_cpu_gb,
            copy_testbed=args.copy_testbed, passthru=passthru,
            agent=args.agent, model=args.model,
            sysbox_milestones=_sysbox_set,
            cpu_pinning_milestones=_cpu_pin_set,
        )
        for repo, mid in tasks
    ]
    if _sysbox_set:
        _n_sysbox = sum(1 for t in task_objs if t.sysbox)
        print(f"  sysbox-runc pool: {_n_sysbox}/{len(task_objs)} task(s) "
              f"(allowlist: {sorted(_sysbox_set)}); the rest use the runc pool")
        if _n_sysbox == 0:
            print("  WARN: --sysbox-milestone matched no task in this run "
                  "(check the mid / repo/mid spelling)")

    sweep_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda t: t.run(), task_objs))
    total = int(time.time() - sweep_start)

    ok = sum(1 for t in task_objs if t.rc == 0)
    print()
    print(f"=== sweep summary [agent={args.agent}] (parallelization-level={args.by}, "
          f"wall-clock {total}s ~ {total // 60}m, workers={workers}) ===")
    print("task\tseconds\tcompleted\tfailed\trc")
    for t in sorted(task_objs, key=lambda t: t.label):
        print(f"{t.label}\t{t.seconds}s\t{t.completed}\t{t.failed}\t{t.rc}")
    print()
    print(f"tasks rc=0: {ok} / {len(tasks)}   |   logs: {log_dir}/<task>.log")
    print(f"run results (dataset-symlinks stripped; EvoClaw outputs only): {run_ws}")
    # ONE eyeball-friendly roll-up of the REAL result (milestones resolved) at the root.
    _write_xrlenv_summary(run_ws, args.by)
    print(f"per-milestone detail: {run_ws}/<repo>__<mid>/<repo>/e2e_trial/*/evaluation/")
    return 0 if ok == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
