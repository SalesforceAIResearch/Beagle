#!/usr/bin/env python3
"""Run the benchmark suite described by ``xrlenv_plugins/benchmarks/tests/integration/benchmarks.yaml``.

    python xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py --profile full   # whole green set per benchmark
    python xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py --profile ci     # a deterministic k-task sample
    python xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py --profile ci --benchmark lhtb,seta --dry-run

This is the benchmark-onboarding integration harness: a config-driven runner over
the ``run_full_sweep.sh`` / ``run_oracle_sweep.py`` sweep contract, launched by
hand for a full green-set sweep or with a small deterministic sample for CI.

Two modes, both running ONLY ``present - EXCLUDE`` (each benchmark's own
``run_full_sweep.sh`` owns its EXCLUDE / blacklist — the single source of
known-failing tasks):

* ``full``   -> the benchmark's ``run_full_sweep.sh`` over its whole GREEN SET
  (present - EXCLUDE — never all present tasks), with its content-retry layer.
* ``sample`` -> read the green set via ``run_full_sweep.sh --list-green``, pick ``k``
  tasks DETERMINISTICALLY (seeded — same set every run; change the seed to rotate),
  then ``run_oracle_sweep.py --tasks <sample>``.

Exit code is 0 iff every selected benchmark's sweep passed, so this is CI-usable.
Covers the 5 harbor/pier benchmarks + swebench_verified (a docker-py drop-in that
now shares the same sweep contract). evoclaw + webarena-infinity have bespoke
entrypoints and are run manually.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[4]  # xrlenv_plugins/benchmarks/tests/integration/ -> repo root
BENCH_DIR = REPO / "xrlenv_plugins" / "benchmarks"
PY = str(REPO / ".venv" / "bin" / "python")
DEFAULT_CONFIG = Path(__file__).resolve().parent / "benchmarks.yaml"
# A task id is a single directory-name token; the ``==> ...`` progress lines the
# scripts print have spaces, so this cleanly separates ids from LIST_GREEN noise.
_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# run_full_sweep.sh AND run_oracle_sweep.py are both uniform flag interfaces across
# all five now — no per-benchmark Python table. Sample mode passes --tasks /
# --max-workers / --jobs-dir / --job-id, plus --retries N when the benchmark's
# `retries` knob (in benchmarks.yaml) is non-null. seta has no --retries flag, so its
# yaml `retries` is null; everything else lives in benchmarks.yaml, nothing hidden here.


def _load_dotenv() -> None:
    """Best-effort: load repo-root .env into os.environ (the sweeps expect
    XRLENV_BENCHMARK_CACHE + CP creds there). Never overwrites an already-set var."""
    env = REPO / ".env"
    if not env.is_file():
        return
    for line in env.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"config not found: {path}") from None
    if not isinstance(cfg, dict) or "benchmarks" not in cfg or "profiles" not in cfg:
        raise SystemExit(f"{path}: expected a mapping with `benchmarks` + `profiles`")
    return cfg


def _effective(cfg: dict[str, Any], profile: str, name: str) -> dict[str, Any]:
    """Merge: defaults <- benchmarks[name] <- profiles[profile] <- profile overrides[name]."""
    prof = cfg["profiles"].get(profile)
    if prof is None:
        raise SystemExit(
            f"unknown profile {profile!r}; have: {', '.join(cfg['profiles'])}"
        )
    eff: dict[str, Any] = {}
    eff.update(cfg.get("defaults") or {})
    eff.update(cfg["benchmarks"][name] or {})
    eff.update({k: v for k, v in prof.items()
                if k not in ("overrides", "only", "max_concurrent_tasks")})
    eff.update((prof.get("overrides") or {}).get(name) or {})
    return eff


def _profile_budget(prof: dict[str, Any], cfg: dict[str, Any]) -> int:
    """The scheduling budget for a profile: its ``max_concurrent_tasks`` if set,
    else the top-level default, else 128.

    Accepts the key at the profile level (canonical) OR inside the profile's
    ``overrides:`` block — a common, natural placement (``overrides`` reads as "what
    this profile overrides"). Honoring both means an operator's value is respected
    rather than silently ignored; the profile level wins if somehow both are present.
    A scalar under ``overrides`` never collides with the per-benchmark dicts there,
    since those are keyed by benchmark name and consumed via ``.get(<name>)``."""
    val = prof.get("max_concurrent_tasks")
    if val is None:
        val = (prof.get("overrides") or {}).get("max_concurrent_tasks")
    if val is None:
        val = cfg.get("max_concurrent_tasks", 128)
    return int(val)


def _select_names(cfg: dict[str, Any], profile: str, benchmark_arg: str | None) -> list[str]:
    """Which benchmarks to run. Precedence: an explicit ``--benchmark`` override >
    the profile's ``only:`` picked set > all benchmarks (config order preserved)."""
    all_names = list(cfg["benchmarks"])
    if benchmark_arg:
        want = [n.strip() for n in benchmark_arg.split(",") if n.strip()]
        unknown = [n for n in want if n not in cfg["benchmarks"]]
        if unknown:
            raise SystemExit(f"unknown benchmark(s): {', '.join(unknown)}; have: {', '.join(all_names)}")
        # `--benchmark ,` / `--benchmark ""` parses to nothing — that must FAIL, not run
        # zero plans and exit 0 with "all selected benchmarks passed" (a false green).
        if not want:
            raise SystemExit(f"--benchmark {benchmark_arg!r} selected no benchmarks; "
                             f"have: {', '.join(all_names)}")
        # Reject duplicates (`--benchmark lhtb,lhtb`): two plans share one benchmark log +
        # one job-id, so the second run's artifacts would collide with the first's and the
        # coverage gate would double-count (audit M3).
        dupes = sorted({n for n in want if want.count(n) > 1})
        if dupes:
            raise SystemExit(f"--benchmark lists duplicate(s): {', '.join(dupes)}")
        return want
    only = (cfg["profiles"].get(profile) or {}).get("only")
    if only:
        unknown = [n for n in only if n not in cfg["benchmarks"]]
        if unknown:
            raise SystemExit(
                f"profile {profile!r} `only` lists unknown benchmark(s): {', '.join(unknown)}"
            )
        return [n for n in all_names if n in only]
    # An empty `benchmarks:` mapping must not silently pass — there is nothing to gate.
    if not all_names:
        raise SystemExit("config has no benchmarks to run (`benchmarks:` is empty).")
    return all_names


def _resolve_run_dir(jobs_dir: str, run_name: str) -> Path:
    """``<jobs_dir>/<run_name>`` with ``run_name`` validated as a SINGLE bare component
    strictly beneath the resolved jobs root.

    An absolute or ``..`` run name would escape — ``Path("/jobs") / "/tmp/x"`` is
    ``Path("/tmp/x")``, and ``..`` climbs out — and ``--overwrite``'s ``shutil.rmtree``
    would then recursively delete an arbitrary directory (audit H5). Both the bare-component
    check AND a resolved strict-child check (which also catches symlinked roots) must pass."""
    if run_name != Path(run_name).name or run_name in ("", ".", ".."):
        raise SystemExit(f"--run-name must be a single path component (got {run_name!r})")
    jobs_root = Path(jobs_dir).expanduser().resolve()
    run_dir = jobs_root / run_name
    if jobs_root not in run_dir.resolve().parents:
        raise SystemExit(f"run dir {run_dir} escapes the jobs dir {jobs_root}")
    return run_dir


def _positive_k(name: str, eff: dict[str, Any]) -> int:
    """The sample size, validated. k<=0 must FAIL: k=0 emits ``--tasks ""`` (drivers read
    an empty selector as their DEFAULT task set) and a negative k hits Python's negative
    slicing and can select almost the whole green set — both silently run the wrong set."""
    k = int(eff.get("k", 5))
    if k <= 0:
        raise SystemExit(f"{name}: sample mode needs k >= 1 (got k={k}); a non-positive k "
                         f"selects an empty or unintended task set.")
    return k


def _list_green(name: str) -> tuple[list[str], int]:
    """The green set (present - EXCLUDE) + the TOTAL present corpus size, via
    `run_full_sweep.sh --list-green --skip-build-cache`. Filters stdout to task-id lines
    (the green set) and parses ``#TOTAL_PRESENT=<n>`` from stderr (the full corpus, before
    EXCLUDE); falls back to the green count if a wrapper doesn't emit it.

    ALWAYS ``--skip-build-cache``: the runner is a READ-ONLY gate over prepared cache/image
    prerequisites (audit M12), so it never builds a cache during planning. The cache must
    already exist — an absent/incomplete one fails loud below with the wrapper's bring-up
    recipe in the error.

    A NON-ZERO exit means the wrapper couldn't even compute the green set (unbuilt/incomplete
    cache, missing registry, a failed completeness/membership check) — we must NOT trust
    task-shaped stdout then, or a partial/garbled listing becomes the "requested set" and a
    subset sweep silently passes. Require rc == 0."""
    script = BENCH_DIR / name / "run_full_sweep.sh"
    cmd = ["bash", str(script), "--list-green", "--skip-build-cache"]
    out = subprocess.run(cmd, env={**os.environ}, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(
            f"{name}: --list-green exited {out.returncode} — cannot trust its output as "
            f"the requested green set.\n--- stderr ---\n{out.stderr[-1500:]}"
        )
    ids = [ln.strip() for ln in out.stdout.splitlines() if _TASK_ID.match(ln.strip())]
    if not ids:
        raise SystemExit(
            f"{name}: --list-green produced no tasks (is the cache built / "
            f"XRLENV_BENCHMARK_CACHE set?)\n--- stderr ---\n{out.stderr[-1500:]}"
        )
    green = sorted(set(ids))
    m = re.search(r"#TOTAL_PRESENT=(\d+)", out.stderr)
    total = int(m.group(1)) if m else len(green)
    return green, total


def _rewards_pass(rewards: dict[str, Any], benchmark: str) -> bool:
    """Apply the BENCHMARK-SPECIFIC pass rule to a ``result.json`` ``rewards`` dict.

    NOT a universal reward rule (audit M18): each benchmark's own
    ``run_oracle_sweep._trial_passes`` gates differently, and a one-size rule silently
    mis-scores. This mirrors each benchmark's published helper — keep them in sync:

    * ``lhtb`` — ``lhtb/_reward_value``: the ``reward`` key if present, else the **max**
      metric (its game verifiers write several diagnostic metrics with no ``reward`` key
      and pass on a positive partial-credit band; an all-values rule would false-red a
      ``{partial: 0, diagnostic_score: 1}`` that LHTB's own gate passes).
    * ``deep_swe`` — ``deep_swe/_reward_value``: ONLY the ``reward`` key; no key ⇒ fail.
    * ``swebench_pro`` — ``swebench_pro/run_oracle_sweep._trial_passes``: ONLY the ``reward``
      key (its ``reward.json`` also carries diagnostic counts such as ``p2p_total``, which is
      legitimately 0 for an instance with no PASS_TO_PASS tests — an all-values rule would
      false-red a resolved oracle).
    * ``seta`` / ``terminal_bench_2_1`` / ``terminalworld`` (and the safe default): EVERY
      metric strictly positive.

    (``swebench_verified`` never reaches here — it reports via ``summary.json``.)
    """
    try:
        if benchmark == "lhtb":
            if "reward" in rewards:
                return float(rewards["reward"]) > 0
            return max(float(v) for v in rewards.values()) > 0
        if benchmark in ("deep_swe", "swebench_pro"):
            return "reward" in rewards and float(rewards["reward"]) > 0
        return all(float(v) > 0 for v in rewards.values())
    except (TypeError, ValueError):
        return False


def _trial_result_passes(result_json: Path, benchmark: str) -> bool:
    """True iff ``result.json`` records a passing trial under ``benchmark``'s own rule.

    Mirrors every benchmark's ``run_oracle_sweep._trial_passes``: a non-null ``exception_info``
    is a HARD fail regardless of reward (audit M18 — a trial that errored, e.g. ``NodeLost``,
    must not read as passing even if a stale ``reward=1`` is present), then the reward
    interpretation is delegated per-benchmark to :func:`_rewards_pass`. Any read/parse error or
    missing rewards => not passing."""
    try:
        j = json.loads(result_json.read_text())
    except (OSError, ValueError):
        return False
    # A recorded exception fails the trial outright — same first check as the benchmark gates.
    if j.get("exception_info") is not None:
        return False
    vr = j.get("verifier_result") or {}
    rewards = vr.get("rewards") if isinstance(vr, dict) else None
    if not rewards or not isinstance(rewards, dict):
        return False
    return _rewards_pass(rewards, benchmark)


def _summary_json_resolved(summary_json: Path) -> set[str]:
    """The resolved instance ids from a SWE-bench ``<job>/summary.json``
    (``{"instances": [{"instance_id": ..., "resolved": bool}, ...]}``). Empty set on
    any read/parse error, or for a Harbor/Pier run (which writes no ``summary.json``).
    This is the second artifact shape the coverage gate understands — SWE-bench does
    NOT write per-task trial dirs, so without this a resolved SWE run reads as 0/N."""
    try:
        j = json.loads(summary_json.read_text())
    except (OSError, ValueError):
        return set()
    rows = j.get("instances") if isinstance(j, dict) else None
    if not isinstance(rows, list):
        return set()
    return {str(r["instance_id"]) for r in rows
            if isinstance(r, dict) and r.get("resolved") and r.get("instance_id")}


def _canonical_task_id(result_json: Path, fallback_dir_name: str) -> str:
    """The REQUESTED task id for a Harbor/Pier trial dir.

    Read from ``result.json["config"]["task"]["path"]`` (basename) — the UNtruncated
    identity harbor/pier record. The trial DIR name is ``<task-id>__<suffix>`` but harbor
    TRUNCATES the ``<task-id>`` part to ~32 chars, so ``rsplit("__", 1)[0]`` of the dir
    name yields a truncated alias for long ids (``tengo-callable-instance-isolation`` ->
    dir ``tengo-callable-instance-isolatio``) that never matches the requested id — the
    coverage gate then falsely reports a passing task as missing and forces a green run RED
    (audit H3). Falls back to the dir-name split only when the path is unreadable."""
    try:
        j = json.loads(result_json.read_text())
        path = ((j.get("config") or {}).get("task") or {}).get("path")
        if path:
            return Path(path).name
    except (OSError, ValueError, AttributeError):
        pass
    return fallback_dir_name.rsplit("__", 1)[0]


def _passing_tasks(run_root: Path, job_id: str, benchmark: str) -> set[str]:
    """The set of task ids that produced a PASSING artifact under ``run_root``.

    Understands BOTH sweep-artifact shapes (benchmark-aware, no per-benchmark config —
    a Harbor run has no ``summary.json``, a SWE run has no valid trial ``result.json``):

    * Harbor/Pier — one trial dir per task at
      ``<jobs-dir>/<job-id>[-<ts>][-retryN]/<task-id>__<suffix>/result.json``. The task id
      is read from the artifact's ``config.task.path`` (NOT the truncated dir name — see
      ``_canonical_task_id``); the trial passes iff ``verifier_result`` records reward>0.
    * SWE-bench — a single ``<jobs-dir>/<job-id>[-<ts>]/summary.json`` with per-instance
      ``resolved`` flags (SWE writes no per-task trial dirs).

    We union across ALL job dirs whose name starts with ``job_id`` so content-retry rounds
    (``…-retryN`` siblings) count: a task passing in ANY round is passing, mirroring the
    sweeps' own retry logic. The runner groups each invocation under a unique ``run_dir``,
    so ``{job_id}*`` matches only THIS run's dirs — never a prior run's stale pass."""
    passing: set[str] = set()
    for job_dir in sorted(run_root.glob(f"{job_id}*")):
        if not job_dir.is_dir():
            continue
        # SWE-bench shape: a single per-instance summary.json.
        passing |= _summary_json_resolved(job_dir / "summary.json")
        # Harbor/Pier shape: one trial dir per task with a verifier reward.
        for trial_dir in sorted(job_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            result_json = trial_dir / "result.json"
            if _trial_result_passes(result_json, benchmark):
                passing.add(_canonical_task_id(result_json, trial_dir.name))
    return passing


def _plan_cost(plan: dict[str, Any]) -> int:
    """A benchmark's scheduling cost = its REAL peak concurrency = min(task-count,
    workers). It runs at most ``workers`` rollouts at once (and never more than its
    task count), so this — not the raw task count — is what it charges against the
    budget. A plan without ``workers`` falls back to its task count (old behavior)."""
    n = int(plan["n_tasks"])
    return min(n, int(plan.get("workers", n)))


def _fmt_elapsed(seconds: float) -> str:
    """Compact elapsed clock: ``43s`` / ``17m04s`` / ``2h07m``."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _benchmark_progress(run_root: Path, job_id: str) -> dict[str, Any]:
    """Live per-benchmark progress from disk (existence globs + dir-name dedup, no
    JSON reads — an approximate live view, not the exactness the coverage gate needs).

    Returns ``{finished, error, active, retried, swe}``:

    * ``finished`` — distinct tasks that reached a GRADED verdict (pass or fail) and are
      NOT currently being re-run. Pier/Harbor: trial dirs with a ``result.json`` (deduped
      by the ``<task>__`` dir prefix). SWE-bench: distinct ``<instance>`` dirs with a
      ``report.json``. A task with a fresh, ungraded attempt dir (a content-retry in
      flight) is subtracted out — it shows as ``active``/remaining, not ``finished``, until
      its re-grade lands (so the row can't read "494 finished" while 4 are still re-running).
    * ``error`` — distinct tasks CURRENTLY infra-errored: a trial dir with an
      ``exception.txt`` and no ``result.json`` in ANY attempt. Disjoint from
      ``finished`` (a task graded on a later attempt is finished, not errored), so
      ``finished + error + remaining == total`` holds at every snapshot. Not
      disk-measurable for SWE-bench (an instance without ``report.json`` is
      indistinguishable in-flight vs errored) → 0 there, flagged by ``swe``.
    * ``active`` — distinct tasks with a STARTED-but-ungraded attempt right now (SWE-bench:
      a ``run_instance.log`` with no sibling ``report.json``; Pier/Harbor: a trial dir with
      neither marker). This is the REAL live-task count — a subset of ``remaining`` — as
      opposed to the scheduler's reserved concurrency budget (which over-counts at the
      content-retry tail, e.g. 32 reserved while only 4 evals are truly live).
    * ``retried`` — cumulative infra-error occurrences so far (every ``exception.txt``),
      a churn diagnostic NOT part of the ``finished/error/remaining`` sum. ``None`` for
      SWE-bench (not measurable).
    * ``swe`` — True for the SWE-bench shape, so the caller can footnote error/retried.

    Best-effort: unreadable dirs are skipped, never raised."""
    reported: set[str] = set()     # tasks with a GRADED verdict in some attempt dir
    errored_raw: set[str] = set()  # tasks with an infra-error marker in some attempt
    active: set[str] = set()       # tasks with a STARTED-but-ungraded attempt (LIVE now)
    retried = 0
    swe = False
    for job_dir in run_root.glob(f"{job_id}*"):
        if not job_dir.is_dir():
            continue
        if (job_dir / "logs" / "run_evaluation").is_dir():     # SWE-bench shape
            swe = True
            # One dir per (run_id, model, instance): run_instance.log is written at
            # START, report.json at grade. A dir with the start marker but no report is
            # a LIVE attempt — the only disk signal for the content-retry phase, where a
            # re-run instance keeps its first-pass report.json but gets a fresh,
            # report-less retry dir under a new run_id. Tracking it lets ``finished``
            # drop the few being re-graded (so the row shows them as remaining/running,
            # not falsely "done") and feeds the true live-task count.
            for inst_dir in job_dir.glob("logs/run_evaluation/*/*/*"):
                if not inst_dir.is_dir():
                    continue
                if (inst_dir / "report.json").is_file():
                    reported.add(inst_dir.name)
                elif (inst_dir / "run_instance.log").is_file():
                    active.add(inst_dir.name)
            continue
        try:
            entries = list(job_dir.iterdir())                  # Pier/Harbor shape
        except OSError:
            continue
        for trial_dir in entries:
            if not trial_dir.is_dir():
                continue
            task = trial_dir.name.rsplit("__", 1)[0]           # <task>__<suffix>
            has_result = (trial_dir / "result.json").is_file()
            has_exc = (trial_dir / "exception.txt").is_file()
            if has_result:
                reported.add(task)
            elif has_exc:
                errored_raw.add(task)
            else:
                active.add(task)                               # trial dir, no verdict yet → live
            if has_exc:
                retried += 1
    # Precedence active > finished > errored: a task with a live (re-)run is in-flight
    # regardless of past attempts; else a graded verdict wins over an infra error. The
    # three sets are disjoint, so finished + error + remaining == total still holds and
    # active ⊆ remaining.
    finished = reported - active
    errored = errored_raw - reported - active
    return {"finished": len(finished), "error": len(errored), "active": len(active),
            "retried": (None if swe else retried), "swe": swe}


def _verify_coverage(name: str, requested: list[str], passing: set[str]) -> str | None:
    """Compare the REQUESTED task set against the tasks that actually produced a passing
    artifact. Returns None if every requested task passed, else a one-line failure message
    naming the missing/failed ids. Extra passing tasks not in the requested set are ignored
    (a wrapper may run TBD tasks alongside GREEN)."""
    want = set(requested)
    missing = sorted(want - passing)
    if missing:
        preview = ", ".join(missing[:20]) + (" …" if len(missing) > 20 else "")
        return (
            f"{name}: {len(missing)}/{len(want)} requested task(s) have NO passing "
            f"result — sweep was partial or failed: {preview}"
        )
    return None


def _sample(green: list[str], k: int, seed: int) -> list[str]:
    """A deterministic k-task sample: seeded shuffle of the sorted green set,
    take k, return sorted (stable --tasks order). k>=len -> the whole set."""
    pool = sorted(green)
    if k >= len(pool):
        return pool
    rng = random.Random(seed)
    rng.shuffle(pool)
    return sorted(pool[:k])


def _full_cmd(name: str, eff: dict[str, Any], job_id: str,
              jobs_dir: str) -> tuple[list[str], dict[str, str]]:
    """run_full_sweep.sh over the whole green set (present - EXCLUDE) — a uniform
    flag interface across all benchmarks. ``--jobs-dir`` groups this run's
    artifacts under the shared run dir. ``--skip-build-cache`` keeps the gate READ-ONLY
    (audit M12): execution never populates the cache — that's a bring-up prerequisite."""
    script = str(BENCH_DIR / name / "run_full_sweep.sh")
    cmd = ["bash", script,
           "--skip-build-cache",
           "--max-workers", str(eff.get("workers", 8)),
           "--content-retries", str(eff.get("content_retries", 2)),
           "--jobs-dir", jobs_dir,
           "--job-id", job_id]
    return cmd, {**os.environ}


def _sample_cmd(name: str, tasks: list[str], eff: dict[str, Any],
                jobs_dir: str, job_id: str) -> list[str]:
    cmd = [PY, str(BENCH_DIR / name / "run_oracle_sweep.py"),
           "--tasks", ",".join(tasks),
           "--max-workers", str(eff.get("workers", 8)),
           "--jobs-dir", jobs_dir, "--job-id", job_id]
    retries = eff.get("retries")   # infra-transient --retries (benchmarks.yaml per-benchmark)
    if retries is not None:
        cmd += ["--retries", str(retries)]
    cr = eff.get("content_retries")   # per-task content-retry: re-run reward=0 flakes (ci gap fix)
    if cr is not None:
        cmd += ["--content-retries", str(cr)]
    return cmd


def _run(cmd: list[str], env: dict[str, str], log_file: Any = None) -> int:
    """Run a sweep; stream to stdout (log_file=None) or capture to a file. Returns rc."""
    if log_file is None:
        print(f"    $ {' '.join(cmd)}", flush=True)
        return subprocess.run(cmd, env=env).returncode
    log_file.write(f"$ {' '.join(cmd)}\n\n")
    log_file.flush()
    return subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT).returncode


def _plan(cfg: dict[str, Any], profile: str, name: str, run_dir: Path,
          seed_override: int | None) -> dict[str, Any]:
    """Resolve one benchmark to a runnable command + its ``n_tasks`` — the number of
    tasks it will run (the sampled count in sample mode; the green-set size in full
    mode). That count is what the parallel scheduler bin-packs against the
    ``max_concurrent_tasks`` budget.

    Every profile is a READ-ONLY gate over prepared cache/image prerequisites (audit M12):
    planning lists with ``--skip-build-cache`` and never builds a cache or an image — the
    runner can't build/push all prerequisite images anyway, so a "full" gate must not
    half-bootstrap. An absent/incomplete cache fails loud (via ``_list_green``) with the
    bring-up recipe; prepare it first (see the README's bring-up sequence)."""
    eff = _effective(cfg, profile, name)
    mode = eff.get("mode")
    if mode not in ("full", "sample"):
        raise SystemExit(f"{name}: profile {profile!r} must set mode: full|sample (got {mode!r})")
    job_id = f"{name}-{profile}"
    if mode == "full":
        green, total = _list_green(name)   # read-only (audit M12): never builds the cache
        cmd, env = _full_cmd(name, eff, job_id, str(run_dir))
        # `requested` (the id list) + `job_id` let the post-run coverage gate confirm
        # every one of these tasks produced a passing artifact under run_dir/<job_id>*/.
        return {"benchmark": name, "mode": mode, "cmd": cmd, "env": env,
                "n_tasks": len(green), "green": len(green), "total": total,
                "requested": green, "job_id": job_id,
                "workers": int(eff.get("workers", 8))}
    k = _positive_k(name, eff)
    seed = int(seed_override if seed_override is not None else eff.get("seed", 0))
    green, total = _list_green(name)
    tasks = _sample(green, k, seed)
    cmd = _sample_cmd(name, tasks, eff, str(run_dir), job_id)
    return {"benchmark": name, "mode": mode, "cmd": cmd, "env": {**os.environ},
            "n_tasks": len(tasks), "green": len(green), "total": total, "k": k,
            "seed": seed, "tasks": tasks, "requested": tasks, "job_id": job_id,
            "workers": int(eff.get("workers", 8))}


def _result_of(plan: dict[str, Any], rc: int, secs: float) -> dict[str, Any]:
    r: dict[str, Any] = {"benchmark": plan["benchmark"], "mode": plan["mode"], "rc": rc,
                         "passed": rc == 0, "seconds": secs, "n_tasks": plan["n_tasks"]}
    if plan["mode"] == "sample":
        r.update({"k": plan["k"], "seed": plan["seed"],
                  "green": plan["green"], "tasks": plan["tasks"]})
    return r


def _apply_coverage(results: list[dict[str, Any]], plans: list[dict[str, Any]],
                    run_dir: Path) -> None:
    """Fold the artifact-coverage gate into each result IN PLACE: a benchmark passes only
    if its sweep exited 0 AND every requested task produced a passing artifact under the
    run dir. Independent of each sweep's own exit-code gate — it catches a sweep that
    (regression) swallowed a failure, or ran only a subset: rc==0 alone is not trusted.
    Both run paths write to run_dir/<job_id>*/, so this single post-pass covers both."""
    for r, plan in zip(results, plans, strict=True):   # same length: results derive from plans
        requested = plan["requested"]
        passing = _passing_tasks(run_dir, plan["job_id"], plan["benchmark"])
        cov = _verify_coverage(plan["benchmark"], requested, passing)
        r["requested"] = len(requested)
        r["passing"] = len(passing)
        r["coverage_error"] = cov
        if cov is not None:
            r["passed"] = False   # rc may have been 0 — the artifacts say otherwise
            print(f"  ✗ coverage  {plan['benchmark']:20} {cov}", flush=True)


_BANNER_TICK_S = 5.0    # TTY: redraw the live-status block this often (the heartbeat)
_HEARTBEAT_S = 30.0     # non-TTY (captured log): append one aggregate line this often

_STATE_LABEL = {"queued": "queued", "running": "running",
                "passed": "✓ passed", "failed": "✗ failed"}


class _LiveStatus:
    """The live per-benchmark status block + aggregate footer.

    TTY: redraws a sticky block (one row per benchmark + a ``Σ`` footer) in place every
    heartbeat; permanent ``▶``/``✓``/``✗`` lines scroll above it. Captured to a file/pipe
    (no cursor control): appends one compact ``Σ`` line every :data:`_HEARTBEAT_S` instead.
    All counts come from disk via :func:`_benchmark_progress` — ``finished + error +
    remaining == total`` per row and in the aggregate; ``retried`` is a churn diagnostic
    outside that sum."""

    def __init__(self, plans: list[dict[str, Any]], run_dir: Path, budget: int,
                 started: float, is_tty: bool) -> None:
        self._plans = plans
        self._run_dir = run_dir
        self._budget = budget
        self._started = started
        self._is_tty = is_tty
        self._state: dict[int, str] = dict.fromkeys(range(len(plans)), "queued")
        self._block_lines = 0     # height of the sticky block on screen (TTY)
        self._last_append = 0.0   # non-TTY: wall-clock of the last aggregate append

    def mark(self, idx: int, state: str) -> None:
        self._state[idx] = state

    def permanent(self, text: str) -> None:
        """A ``▶``/``✓``/``✗`` scroll-back line. On a TTY, erase the sticky block first so
        the line lands above where the block will next redraw."""
        if self._is_tty and self._block_lines:
            sys.stdout.write(f"\033[{self._block_lines}A\033[J")
            self._block_lines = 0
        print(text, flush=True)

    def heartbeat(self, in_flight: int, *, final: bool = False) -> None:
        if self._is_tty:
            self._redraw(self._block(in_flight))
            return
        now = time.time()
        if final or now - self._last_append >= _HEARTBEAT_S:
            self._last_append = now
            _, agg = self._collect()
            print(f"[{_fmt_elapsed(now - self._started)}] Σ total {agg['total']} | "
                  f"finished {agg['finished']} | remaining {agg['remaining']} | "
                  f"error {agg['error']} | retried {agg['retried']} | "
                  f"{agg['active']} running",
                  flush=True)

    def finish(self, in_flight: int) -> None:
        """Final refresh so the last numbers land (and, on a TTY, the block stays)."""
        self.heartbeat(in_flight, final=True)

    def _collect(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        agg = {"total": 0, "finished": 0, "error": 0, "active": 0,
               "retried": 0, "any_swe": False}
        for i, plan in enumerate(self._plans):
            total = int(plan["n_tasks"])
            agg["total"] += total
            state = self._state[i]
            jid = plan.get("job_id")
            if state == "queued" or not jid:
                rows.append({"name": plan["benchmark"], "state": state, "counts": None})
                continue
            p = _benchmark_progress(self._run_dir, jid)
            fin = min(int(p["finished"]), total)
            err = min(int(p["error"]), total - fin)
            act = min(int(p["active"]), total - fin - err)   # active ⊆ remaining
            agg["finished"] += fin
            agg["error"] += err
            agg["active"] += act
            agg["retried"] += int(p["retried"] or 0)
            agg["any_swe"] = agg["any_swe"] or p["swe"]
            rows.append({"name": plan["benchmark"], "state": state, "total": total,
                         "finished": fin, "error": err, "remaining": total - fin - err,
                         "active": act, "retried": p["retried"], "swe": p["swe"],
                         "counts": True})
        agg["remaining"] = agg["total"] - agg["finished"] - agg["error"]
        return rows, agg

    def _block(self, in_flight: int) -> list[str]:
        rows, agg = self._collect()
        el = _fmt_elapsed(time.time() - self._started)
        # "running" = REAL live tasks (disk markers); "reserved" = the scheduler's
        # concurrency reservation (Σ min(n_tasks, workers) over running benchmarks),
        # which legitimately exceeds live tasks at a content-retry tail.
        lines = [f"  ── live status · {el} elapsed · {agg['active']} running · "
                 f"{in_flight}/{self._budget} reserved " + "─" * 14]
        for r in rows:
            if r["counts"] is None:
                lines.append(f"  {r['name']:<20} {_STATE_LABEL[r['state']]}")
                continue
            err = "0*" if r["swe"] else str(r["error"])
            ret = "—*" if r["swe"] else str(r["retried"])
            lines.append(
                f"  {r['name']:<20} {_STATE_LABEL[r['state']]:<9}| total {r['total']:>4} | "
                f"finished {r['finished']:>4} | remaining {r['remaining']:>4} | "
                f"error {err:>3} | retried {ret:>3}")
        lines.append("  " + "─" * 80)
        lines.append(
            f"  Σ ALL | total {agg['total']} | finished {agg['finished']} | "
            f"remaining {agg['remaining']} | error {agg['error']} | "
            f"retried {agg['retried']} | {agg['active']} running")
        if agg["any_swe"]:
            lines.append("  * swebench: error/retried not disk-separable mid-run. "
                         "running = live evals; a content-retry re-run shows as running "
                         "(not finished) until it re-grades.")
        return lines

    def _redraw(self, lines: list[str]) -> None:
        cols = shutil.get_terminal_size((120, 24)).columns
        out = []
        if self._block_lines:
            out.append(f"\033[{self._block_lines}A")   # up to the block's first line
        out.append("\033[J")                            # clear cursor→end of screen
        out.append("\n".join(ln[:cols - 1] for ln in lines))
        out.append("\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self._block_lines = len(lines)


def _schedule(plans: list[dict[str, Any]], budget: int, run_dir: Path,
              profile: str) -> list[dict[str, Any]]:
    """Run plans concurrently, admitting them IN ORDER while the TOTAL REAL
    concurrency (summed across the running benchmarks) stays <= ``budget``. Each
    benchmark's cost is min(n_tasks, workers) — its actual peak rollout concurrency,
    NOT its task count — so several benchmarks pack into ``max_concurrent_tasks`` and
    run at once (a 500-task/8-worker sweep costs 8, not 500, and overlaps the rest
    instead of running alone at the end). A plan whose cost exceeds the whole budget
    runs by itself (nothing deadlocks). Each plan streams to
    ``<run_dir>/<name>-<profile>.log`` (interleaved live output would be unreadable).
    Liveness (:class:`_LiveStatus`): on a TTY a per-benchmark status block redraws in
    place every ~5 s; when output is captured (pipe/file) one aggregate line is
    appended every ~30 s. Returns results in plan order."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    started = time.time()
    status = _LiveStatus(plans, run_dir, budget, started, is_tty)
    tick = _BANNER_TICK_S if is_tty else _HEARTBEAT_S

    results: dict[int, dict[str, Any]] = {}
    pending = list(enumerate(plans))
    running: dict[Any, tuple[int, int]] = {}
    in_flight = 0

    def _one(idx: int, plan: dict[str, Any]) -> tuple[int, int, float, Path]:
        log_path = run_dir / f"{plan['benchmark']}-{profile}.log"
        t0 = time.time()
        with open(log_path, "w") as lf:
            lf.write(f"$ {' '.join(plan['cmd'])}\n\n")
            lf.flush()
            rc = subprocess.run(plan["cmd"], env=plan["env"],
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        return idx, rc, round(time.time() - t0, 1), log_path

    with ThreadPoolExecutor(max_workers=len(plans)) as pool:
        while pending or running:
            while pending:
                idx, plan = pending[0]
                n_tasks = int(plan["n_tasks"])
                cost = _plan_cost(plan)   # real concurrency: min(n_tasks, workers)
                if in_flight == 0 or in_flight + cost <= budget:
                    pending.pop(0)
                    in_flight += cost
                    status.mark(idx, "running")
                    status.permanent(f"  ▶ start   {plan['benchmark']:<20} {n_tasks:>4} task(s)")
                    running[pool.submit(_one, idx, plan)] = (idx, cost)
                else:
                    break
            # timeout so the live status refreshes even when nothing finishes
            done, _ = wait(list(running), timeout=tick, return_when=FIRST_COMPLETED)
            if not done:
                status.heartbeat(in_flight)
                continue
            for fut in done:
                idx, cost = running.pop(fut)
                in_flight -= cost
                _idx, rc, secs, log_path = fut.result()
                plan = plans[idx]
                status.mark(idx, "passed" if rc == 0 else "failed")
                mark = "✓ PASS" if rc == 0 else "✗ FAIL"
                status.permanent(f"  {mark}  {plan['benchmark']:<20} {secs:>6.0f}s  "
                                 f"log: {log_path.name}")
                results[idx] = _result_of(plan, rc, secs)
    status.finish(in_flight)
    return [results[i] for i in range(len(plans))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="profile in benchmarks.yaml (e.g. full, ci)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--benchmark", default=None,
                    help="comma-separated subset of the config's benchmarks (default: all)")
    ap.add_argument("--jobs-dir", default=str(REPO / "tmp"),
                    help="parent artifact root (default: ./tmp); the run is grouped "
                         "under <jobs-dir>/<run-name>/")
    ap.add_argument("--run-name", default=None,
                    help="group this whole run's artifacts under <jobs-dir>/<run-name>/ "
                         "(default: <profile>-<timestamp>-<pid><rand>, always unique)")
    ap.add_argument("--overwrite", action="store_true",
                    help="if the run dir already exists, clear it first (default: refuse, "
                         "so a reused dir can't feed stale artifacts to the coverage gate)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the deterministic-sample seed for all benchmarks")
    ap.add_argument("--max-concurrent-tasks", type=int, default=None,
                    help="parallel-scheduling task budget across benchmarks (default: "
                         "benchmarks.yaml `max_concurrent_tasks`, else 128). Benchmarks run "
                         "concurrently while the sum of their task counts stays <= this; a "
                         "benchmark bigger than the budget runs alone. 0 = sequential (live-streamed).")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would run (+ the sampled task ids) without executing")
    args = ap.parse_args()

    _load_dotenv()
    # Fail loud if the caller still uses the retired cache env var/path (renamed 2026-07-31:
    # XRLENV_HARBOR_CACHE -> XRLENV_BENCHMARK_CACHE, .../xrlenv_harbor_cache ->
    # .../xrlenv_benchmark_cache) — a stale env would gate against a moved/absent cache and
    # give unreliable results. Checked here (after .env load) so it fires before any benchmark.
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env
    guard_legacy_cache_env()
    cfg = _load_config(args.config.resolve())

    names = _select_names(cfg, args.profile, args.benchmark)

    # Group this invocation's artifacts under one UNIQUE run dir. The coverage gate globs
    # run_dir/<job-id>*/, so a REUSED run dir would let a prior run's stale passing artifacts
    # satisfy this run (audit M3). The default name carries pid+random so two runs in the
    # same second don't collide; an explicit --run-name that already exists is refused unless
    # --overwrite (which clears it first).
    if args.run_name:
        run_name = args.run_name
    else:
        token = f"{os.getpid()}-{random.randrange(16**4):04x}"
        run_name = f"{args.profile}-{_dt.datetime.now():%Y-%m-%d_%H-%M-%S}-{token}"
    run_dir = _resolve_run_dir(args.jobs_dir, run_name)   # validated strict child (audit H5)
    if not args.dry_run:
        if run_dir.exists():
            if not args.overwrite:
                raise SystemExit(
                    f"run dir already exists: {run_dir}\n  refusing to reuse it — the "
                    f"coverage gate would count a prior run's artifacts. Pass --overwrite to "
                    f"clear it, or --run-name for a fresh label.")
            shutil.rmtree(run_dir)   # safe: run_dir is a validated strict child of jobs_dir
        run_dir.mkdir(parents=True, exist_ok=False)
    print(f"run dir: {run_dir}", flush=True)

    # Parallel-scheduling budget: CLI > profile (level or `overrides:`) > top-level > 128.
    prof = cfg["profiles"].get(args.profile) or {}
    budget = int(args.max_concurrent_tasks) if args.max_concurrent_tasks is not None \
        else _profile_budget(prof, cfg)

    # Plan every benchmark upfront — resolves green sets + task counts, and fails fast
    # if a cache is missing before anything is launched.
    print(f"planning {len(names)} benchmark(s)...", flush=True)
    plans = [_plan(cfg, args.profile, name, run_dir, args.seed) for name in names]
    for p in plans:
        line = f"  {p['benchmark']:20} {p['mode']:6} {p['n_tasks']:>4} task(s)"
        if p["mode"] == "sample":
            line += f"  (sampled from green={p['green']}/{p.get('total', p['green'])}, k={p['k']}, seed={p['seed']} -> {', '.join(p['tasks'])})"
        print(line, flush=True)

    if args.dry_run:
        print(f"\n=== DRY-RUN (max_concurrent_tasks={budget}) ===")
        for p in plans:
            print(f"    $ {' '.join(p['cmd'])}")
        return 0

    # Run: parallel (gated by the task budget) when there's >1 benchmark and budget>0;
    # otherwise sequential with live-streamed output.
    if budget > 0 and len(plans) > 1:
        print(f"\nrunning {len(plans)} benchmark(s) in parallel — up to {budget} concurrent "
              f"tasks TOTAL across all benchmarks (max_concurrent_tasks), NOT per benchmark. "
              f"Live output streams to per-benchmark logs under the run dir; a single liveness "
              f"banner refreshes in place below every {int(_BANNER_TICK_S)}s (when output is "
              f"captured to a file/pipe, a progress line is appended every "
              f"{int(_HEARTBEAT_S)}s instead).", flush=True)
        results = _schedule(plans, budget, run_dir, args.profile)
    else:
        results = []
        for p in plans:
            print(f"\n=== {p['benchmark']}  [{args.profile}: {p['mode']}] ===", flush=True)
            t0 = time.time()
            rc = _run(p["cmd"], p["env"])
            results.append(_result_of(p, rc, round(time.time() - t0, 1)))

    # Independently verify every requested task produced a passing artifact — a sweep
    # can't pass on exit code alone (catches a partial/failed sweep). Flips `passed`
    # to False on any coverage gap, so the summary + exit code reflect it.
    _apply_coverage(results, plans, run_dir)

    # ── summary + exit code ──────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\n=== SUMMARY ===")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        # [passing/requested]: requested = green-set size (full) or k (sample); passing =
        # tasks with a positive-reward artifact. A mismatch is a partial/failed sweep.
        cov = f" [{r.get('passing', 0)}/{r.get('requested', r['n_tasks'])}]"
        extra = f" k={r['k']}" if r["mode"] == "sample" else ""
        print(f"  {r['benchmark']:20} {r['mode']:6} {mark}{cov}  ({r['seconds']}s{extra})")

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / f"benchmarks-{args.profile}-summary.json"
    summary_path.write_text(json.dumps(
        {"profile": args.profile, "run_name": run_name, "run_dir": str(run_dir),
         "max_concurrent_tasks": budget, "results": results}, indent=2))
    print(f"\nrun dir: {run_dir}\nsummary: {summary_path}")
    failed = [r["benchmark"] for r in results if not r.get("passed")]
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("all selected benchmarks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
