"""Score a candidate on benchmarks the legacy runner physically cannot run.

WHY THIS EXISTS
    DeepSWE sets ``allow_internet=false`` on every task. The legacy pipeline cannot honour
    that: its vendored xrlenv has no pier support, so the trial dies at network setup. beagle
    drives DeepSWE through pier's *phased* network instead -- INSTALL with egress open against
    an allowlist (clone + npm build), then RUN restricted to the LLM gateway alone. That is the
    only path on which DeepSWE runs at all, and it is the honest one: it gives the agent exactly
    the isolation the task asked for rather than handing it the internet to earn a pass.

    So the mixture gate has two eval backends, not one. This module is the second: hand it a
    benchmark, a task list and the candidate's git ref, and it returns per-task pass rates in
    the same shape ``codingbench_eval.run_subset_sampled`` returns. ``multibench`` picks the
    backend per benchmark and does not otherwise care.

WHY A SUBPROCESS RATHER THAN AN IMPORT
    beagle lives in its own virtualenv with its own pinned ``datacurve-pier``, deliberately
    kept in lockstep with its vendored xrlenv. Importing it into the driver's interpreter would
    couple two dependency sets that are pinned against different things, and the first version
    skew would surface as a mystery at rollout time. The CLI is a stable, documented seam --
    ``beagle evaluate --config <yaml>`` -- and a subprocess boundary is exactly the right amount
    of coupling.

WHAT IS DELIBERATELY NOT HERE
    No fallback to the legacy runner. If beagle cannot score a benchmark, the caller gets an
    empty result and ``multibench`` omits that benchmark from the mixture, which is the same
    thing it does for any unmeasurable benchmark. Silently scoring DeepSWE through a runner
    that cannot isolate it would produce numbers that look fine and mean nothing.
"""

from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

DEFAULT_ROOT = str(pathlib.Path(__file__).resolve().parents[5])  # repo root; override via DARWINX_GATE_BEAGLE_ROOT
#: Benchmarks routed through beagle rather than the legacy runner. DeepSWE is the only one that
#: *must* be, but the list is a knob so adding the next filtered-egress benchmark is config.
DEFAULT_BENCHMARKS = "deep-swe"
DEFAULT_RUNTIME = "xrlenv-cluster"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_PROVIDER = "llm-gateway-express-local-proxy"
DEFAULT_PARALLELISM = 3
DEFAULT_MAX_TURNS = 250
DEFAULT_TIMEOUT_SEC = 3600
#: Wall clock for the whole subprocess. DeepSWE trials run ~24 min each, so a subset of n tasks
#: at parallelism p needs roughly n/p * 24 min; this is that with generous headroom, and it is
#: a guard against a hung job rather than a budget.
DEFAULT_JOB_TIMEOUT_SEC = 6 * 3600

#: monet CLI flags. These REPLACE beagle's defaults wholesale, which is required rather than
#: cosmetic: beagle's defaults target the pinned monet build 20260805, and monet_code@main is a
#: different CLI that rejects both ``--effort`` and ``--permissive-auto-approve``. Either flag
#: yields a trial that *completes* with reward=0 and an empty agent stream -- a failure that
#: reads as a legitimate benchmark result unless someone opens monet.stderr.log. Reasoning
#: effort is injected as ``reasoning_effort=medium`` by the gateway sanitizer on every request,
#: which is also how the baselines get medium, so no ``effort:`` key belongs here.
MONET_ARGS = ("--all-permissions", "--no-monet-md", "--output-format", "stream-json")

#: Same markers as calibrate_baselines.py and beagle/eval/validation.py. An infra failure must
#: leave the denominator rather than count as a loss, on both the baseline and the candidate
#: side; a rule applied on one side only produces a difference that is pure bookkeeping.
UNMEASURED_MARKERS = (
    "ConnectionError", "ConnectError", "ProtocolError", "RemoteDisconnected",
    "ImageNotFound", "ContainerError", "DockerException", "PermissionError",
    "HTTP 401", "HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504",
    "gateway", "Gateway", "tunnel", "APIConnectionError", "RateLimit",
    "InstallError", "SetupError", "harness", "Harness",
    "unknown option",   # a CLI mismatch is our bug, never the agent failing the task
)


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v.strip() else default


def beagle_root() -> pathlib.Path:
    return pathlib.Path(_env("DARWINX_GATE_BEAGLE_ROOT", DEFAULT_ROOT))


def routed_benchmarks() -> frozenset[str]:
    raw = _env("DARWINX_GATE_BEAGLE_BENCHMARKS", DEFAULT_BENCHMARKS)
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def available() -> bool:
    """True when a usable beagle install is present.

    Checked rather than assumed: the driver runs on hosts that may not have beagle at all, and
    the correct behaviour there is to omit the benchmark, not to crash a campaign.
    """
    return (beagle_root() / ".venv" / "bin" / "beagle").is_file()


def handles(benchmark: str) -> bool:
    return benchmark in routed_benchmarks() and available()


@dataclass
class BridgeResult:
    """Per-task pass rates, in the shape codingbench_eval returns.

    ``rates`` maps task -> (pass_rate, n_measured_samples). ``unmeasured`` counts trials that
    were thrown away as infrastructure failures, and is reported rather than folded in so a
    caller can tell "the agent failed" from "we failed to measure".
    """

    rates: dict[str, tuple[float, int]] = field(default_factory=dict)
    unmeasured: int = 0
    run_dirs: list[str] = field(default_factory=list)
    error: str | None = None

    def __bool__(self) -> bool:
        return bool(self.rates)


def _base_task(task_name: str, trial_name: str) -> str:
    """The task id as the mixture spec spells it.

    beagle reports ``task_name`` as ``<owner>/<task>`` and ``trial_name`` as ``<task>__<rand>``.
    The spec holds the bare task, so strip both decorations; prefer task_name because the trial
    suffix is only separated by a convention.
    """
    if task_name:
        return task_name.rsplit("/", 1)[-1]
    return trial_name.rsplit("__", 1)[0] if "__" in trial_name else trial_name


def _is_unmeasured(trial: dict) -> bool:
    exc = trial.get("exception_info")
    if exc:
        return True
    agent = (trial.get("agent_result") or {}).get("metadata") or {}
    err = str(agent.get("error") or "")
    if err and any(m in err for m in UNMEASURED_MARKERS):
        return True
    # No verifier block at all means the trial never reached grading -- infrastructure, not a
    # wrong patch. A graded zero, by contrast, has rewards and is a real loss.
    return (trial.get("verifier_result") or {}).get("rewards") is None


def _reward(trial: dict) -> float | None:
    rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
    r = rewards.get("reward")
    try:
        return float(r)
    except (TypeError, ValueError):
        return None


def parse_run(run_dir: pathlib.Path, benchmark: str) -> tuple[dict[str, list[float]], int]:
    """Read one beagle run into task -> [reward, ...] plus an unmeasured count.

    Walks the per-trial ``result.json`` rather than the job-level aggregate: the aggregate keys
    rewards by *value* (reward -> [trial names]), which is lossy to invert and silently wrong
    the moment two tasks share a reward. The per-trial files are the source of truth.
    """
    per_task: dict[str, list[float]] = {}
    unmeasured = 0
    job = run_dir / benchmark
    if not job.is_dir():
        return per_task, unmeasured
    for trial_json in sorted(job.glob("*/result.json")):
        try:
            trial = json.loads(trial_json.read_text())
        except (OSError, json.JSONDecodeError):
            unmeasured += 1
            continue
        task = _base_task(trial.get("task_name") or "", trial.get("trial_name") or "")
        if _is_unmeasured(trial):
            unmeasured += 1
            continue
        r = _reward(trial)
        if r is None:
            unmeasured += 1
            continue
        per_task.setdefault(task, []).append(r)
    return per_task, unmeasured


def _write_config(
    path: pathlib.Path,
    *,
    benchmark: str,
    tasks: list[str],
    repo_url: str,
    ref: str,
    run_dir: pathlib.Path,
    run_name: str,
) -> None:
    # Hand-rolled rather than yaml.safe_dump: the driver's interpreter is not guaranteed to have
    # pyyaml, this file is small and fully known, and a literal template is easier to diff
    # against the checked-in configs the baselines used.
    lines = [
        "run:",
        f"  dir: {run_dir}",
        f"  name: {run_name}",
        f"  runtime: {_env('DARWINX_GATE_BEAGLE_RUNTIME', DEFAULT_RUNTIME)}",
        f"  parallelism: {_env('DARWINX_GATE_BEAGLE_PARALLELISM', str(DEFAULT_PARALLELISM))}",
        "agent:",
        "  harness:",
        "    name: monet",
        "    version: 20260805",
        "    source:",
        f"      repo: {repo_url}",
        f"      ref: {ref}",
        "      token_env: GH_TOKEN",
        "  model:",
        f"    name: {_env('DARWINX_GATE_BEAGLE_MODEL', DEFAULT_MODEL)}",
        f"  provider: {_env('DARWINX_GATE_BEAGLE_PROVIDER', DEFAULT_PROVIDER)}",
        f"  max_turns: {_env('DARWINX_GATE_BEAGLE_MAX_TURNS', str(DEFAULT_MAX_TURNS))}",
        "  forward_env:",
        "    - LLM_GATEWAY_EXPRESS_API_KEY",
        "    - LLM_GATEWAY_EXPRESS_API_KEY_LIST",
        "    - LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL",
        f"  timeout: {_env('DARWINX_GATE_BEAGLE_TIMEOUT', str(DEFAULT_TIMEOUT_SEC))}",
        "  extra_args:",
        "    monet_args:",
    ]
    lines += [f"      - {a}" for a in MONET_ARGS]
    lines += ["data:", f"  - benchmark: {benchmark}", "    tasks:"]
    lines += [f"      - {t}" for t in tasks]
    path.write_text("\n".join(lines) + "\n")


def run_subset(
    benchmark: str,
    task_names: list[str],
    *,
    repo_url: str,
    ref: str,
    k_samples: int = 1,
    run_dir: str | os.PathLike | None = None,
    tee_log_path: str | os.PathLike | None = None,
    job_timeout_sec: int | None = None,
) -> BridgeResult:
    """Score ``ref`` on ``task_names`` of ``benchmark`` through beagle.

    ``k_samples`` repeats the whole subset k times as separate runs, because beagle's evaluate
    has no per-task sample count. k=1 is what the mixture gate uses.
    """
    if not task_names:
        return BridgeResult(error="no tasks requested")
    if not available():
        return BridgeResult(error=f"no beagle install at {beagle_root()}")
    absent = missing_env()
    if absent:
        # Fail before spending a subprocess and 20 seconds per trial on something that cannot
        # work. Without this the symptom is an empty measurement, and the cause is buried in a
        # traceback inside a run directory nobody thinks to look in.
        return BridgeResult(error="cluster env incomplete, refusing to run: "
                                  + ", ".join(absent) + " unset")

    root = beagle_root()
    exe = root / ".venv" / "bin" / "beagle"
    out_root = pathlib.Path(run_dir) if run_dir else root / "tmp" / "bridge"
    out_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    # beagle's HarborCache reads this. Without it DeepSWE resolves zero tasks and the run
    # "succeeds" having measured nothing -- which would look like a clean empty result.
    env.setdefault("XRLENV_BENCHMARK_CACHE", os.path.expanduser("~/.cache/xrlenv/benchmark_cache"))

    accum: dict[str, list[float]] = {}
    unmeasured = 0
    dirs: list[str] = []
    last_error: str | None = None

    for sample in range(max(1, int(k_samples))):
        stamp = f"{int(time.time())}-{sample}"
        run_name = f"bridge-{benchmark}-{stamp}"
        with tempfile.TemporaryDirectory() as td:
            cfg = pathlib.Path(td) / "bridge.yaml"
            _write_config(
                cfg, benchmark=benchmark, tasks=list(task_names), repo_url=repo_url,
                ref=ref, run_dir=out_root, run_name=run_name,
            )
            cmd = [str(exe), "evaluate", "--config", str(cfg)]
            try:
                proc = subprocess.run(
                    cmd, cwd=str(root), env=env, capture_output=True, text=True,
                    timeout=job_timeout_sec or DEFAULT_JOB_TIMEOUT_SEC,
                )
                output = (proc.stdout or "") + (proc.stderr or "")
                if proc.returncode != 0:
                    last_error = f"beagle evaluate rc={proc.returncode}"
            except subprocess.TimeoutExpired:
                output = f"beagle evaluate timed out after {job_timeout_sec} s"
                last_error = output
            except OSError as e:                      # noqa: BLE001 - report, never crash the loop
                output = f"failed to launch {shlex.join(cmd)}: {e}"
                last_error = output

        if tee_log_path:
            try:
                with open(tee_log_path, "a") as fh:
                    fh.write(f"\n=== beagle bridge {benchmark} sample={sample} ===\n{output}\n")
            except OSError:
                pass

        # Find the run beagle actually stamped: it appends its own timestamp to run.name.
        matches = sorted(out_root.glob(f"{run_name}*"), key=lambda p: p.stat().st_mtime)
        if not matches:
            continue
        got, unm = parse_run(matches[-1], benchmark)
        dirs.append(str(matches[-1]))
        unmeasured += unm
        for task, rewards in got.items():
            accum.setdefault(task, []).extend(rewards)

    rates = {
        task: (sum(1.0 for r in rs if r >= 1.0) / len(rs), len(rs))
        for task, rs in accum.items() if rs
    }
    error = None
    if not rates:
        # "no measurable trials" on its own is useless, and worse than useless in a campaign:
        # the caller omits the benchmark and the log says nothing about whether the cluster
        # rejected our token, the corpus was missing, or the agent simply never started. The
        # reason is sitting in the run directory, so put it in the message.
        error = last_error or "no measurable trials"
        why = _first_failure(dirs, benchmark)
        if why:
            error = f"{error}: {why}"
    return BridgeResult(rates=rates, unmeasured=unmeasured, run_dirs=dirs, error=error)


def _first_failure(run_dirs: list[str], benchmark: str) -> str | None:
    """The most specific failure reason available from a run that measured nothing.

    Prefers the trial's own ``exception.txt`` (beagle writes the full traceback there) and falls
    back to the structured ``exception_info``. Returns the last line of the traceback, which is
    the exception itself rather than the frames leading to it.
    """
    for run_dir in reversed(run_dirs):
        job = pathlib.Path(run_dir) / benchmark
        if not job.is_dir():
            continue
        for exc_file in sorted(job.glob("*/exception.txt")):
            try:
                lines = [ln.strip() for ln in exc_file.read_text().splitlines() if ln.strip()]
            except OSError:
                continue
            if lines:
                return lines[-1][:300]
        for trial_json in sorted(job.glob("*/result.json")):
            try:
                info = json.loads(trial_json.read_text()).get("exception_info")
            except (OSError, json.JSONDecodeError):
                continue
            if info:
                return str(info)[:300]
    return None


#: Cluster env the bridge needs inherited from its caller. Checked so a missing token produces
#: one clear line instead of a run that "succeeds" having measured nothing -- which is what the
#: first live call did: XRLENV_CONSUMER_TOKEN was absent, every trial died UNAUTHENTICATED at
#: environment setup, and the only visible symptom was an empty result.
REQUIRED_ENV = ("XRLENV_GRPC_HOST", "XRLENV_CONSUMER_TOKEN")


def missing_env() -> list[str]:
    return [k for k in REQUIRED_ENV if not os.environ.get(k, "").strip()]


__all__ = [
    "MONET_ARGS", "UNMEASURED_MARKERS", "REQUIRED_ENV", "missing_env",
    "available", "handles", "beagle_root", "routed_benchmarks",
    "BridgeResult", "parse_run", "run_subset",
]
