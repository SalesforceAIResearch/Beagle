"""Data layer for the results dashboard — parse each run's native artifacts into a compact summary.

The dashboard (``dashboard.py``) must stay cheap, but the *accurate* token counts (with the cache
split) live only in each trial's raw agent stream. So this module does the one heavy pass: for every
trial it runs beagle's OWN usage parser (``parse_opencode_usage`` / ``parse_monet_usage`` / mini's
``_parse_trajectory`` — the same parsers the rollout pipeline uses, so numbers match a fresh run),
reads the reward + error from ``result.json``, and writes a per-run ``summary.json``. The app then
reads those summaries (fast) and only touches a raw trajectory when you open the trajectory viewer.

Token source note: harbor's ``run.json`` historically zeroed the cache split (a shim bug, now fixed
for new runs) and the ATIF ``trajectory.json`` aggregates inconsistently across harnesses — so the
per-agent parser is the canonical source here, not either of those.
"""
from __future__ import annotations

import importlib
import json
import statistics
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import yaml
from typing import Any, Callable

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
_SKIP_DIRS = {"archive-to-delete"}
_TOKEN_KEYS = ("prompt", "completion", "input_uncached", "cache_read", "cache_write", "total")
#: Bump when summary.json's shape changes so stale caches auto-rebuild (see load_summaries).
_SCHEMA = 4   # bumped: `version` is now <version>_<ref6> (cached ones rebuild)

# $/1M tokens: fresh input / cached-read input / output. ESTIMATES — internal gateway models have no
# public price; edit here or override live in the app's sidebar. "*" is the fallback.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "gpt-5.6-sol": {"input": 1.25, "cached": 0.125, "output": 10.00},
    "*":           {"input": 1.25, "cached": 0.125, "output": 10.00},
}

# harness name (normalized, '-'/'_' folded) → (module, fn, stream filename, kind). kind "usage" returns
# a beagle Usage; "mini" returns (token_dict, n_turns). New agents: add a row (or fall through to ATIF).
_PARSERS: dict[str, tuple[str, str, str, str]] = {
    "opencode": ("beagle.agents.opencode._helpers", "parse_opencode_usage", "opencode.stream.jsonl", "usage"),
    "monet":    ("beagle.agents.monet._helpers", "parse_monet_usage", "monet.stream.jsonl", "usage"),
    "mini_swe": ("beagle.agents.mini_swe", "_parse_trajectory", "mini.traj.json", "mini"),
}
_parser_cache: dict[str, Callable[[str], Any]] = {}


def _parser(module: str, fn: str) -> Callable[[str], Any]:
    key = f"{module}:{fn}"
    if key not in _parser_cache:
        _parser_cache[key] = getattr(importlib.import_module(module), fn)
    return _parser_cache[key]


def _norm_tokens(d: dict[str, Any] | None) -> dict[str, int]:
    d = d or {}
    out = {k: int(d.get(k) or 0) for k in _TOKEN_KEYS}
    if not out["total"]:
        out["total"] = out["prompt"] + out["completion"]
    return out


def _atif_tokens(agent_dir: Path) -> dict[str, int]:
    """Fallback for a harness with no registered parser: ATIF final_metrics (no read/write split)."""
    try:
        fm = json.loads((agent_dir / "trajectory.json").read_text()).get("final_metrics") or {}
    except (OSError, ValueError):
        return _norm_tokens(None)
    prompt = int(fm.get("total_prompt_tokens") or 0)
    cached = min(int(fm.get("total_cached_tokens") or 0), prompt)
    return _norm_tokens({"prompt": prompt, "completion": int(fm.get("total_completion_tokens") or 0),
                         "input_uncached": prompt - cached, "cache_read": cached})


def trial_tokens(agent_dir: Path, harness: str) -> dict[str, int]:
    """Cache-split token dict for one trial, via the harness's canonical beagle parser."""
    spec = _PARSERS.get((harness or "").replace("-", "_"))
    if spec:
        module, fn, fname, kind = spec
        f = agent_dir / fname
        if f.exists():
            try:
                raw = f.read_text()
                if kind == "usage":
                    return _norm_tokens(_parser(module, fn)(raw).to_token_counts())
                return _norm_tokens(_parser(module, fn)(raw)[0])  # mini: (tokens, n_turns)
            except Exception:  # noqa: BLE001 — a broken stream shouldn't sink the summary
                pass
    return _atif_tokens(agent_dir)


def add_tokens(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in _TOKEN_KEYS}


# ── agent-only latency ───────────────────────────────────────────────────────────────────────────
def _parse_ts(s: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError, TypeError):
        return None


def agent_seconds(d: dict[str, Any]) -> float | None:
    """Wall seconds spent in the AGENT-EXECUTION phase for one trial — excludes environment setup,
    agent setup, and the verifier. Handles both result.json layouts: harbor/pier's top-level
    ``agent_execution`` and swe-bench's ``timing.agent_execution``. None if the timing is absent."""
    ae = d.get("agent_execution") or (d.get("timing") or {}).get("agent_execution")
    if not isinstance(ae, dict):
        return None
    a, b = _parse_ts(ae.get("started_at")), _parse_ts(ae.get("finished_at"))
    if a is None or b is None or b < a:
        return None
    return b - a


# ── error taxonomy ───────────────────────────────────────────────────────────────────────────────
def classify_error(err: str | None, resolved: bool) -> str:
    """A coarse bucket for the error-analysis tab."""
    if not err:
        return "solved" if resolved else "unsolved"
    e = err.lower()
    if "git clone failed" in e or "invalid username or token" in e or "rpc failed" in e:
        return "clone/auth"
    if "unexpected server error" in e or "connect_error" in e or "502" in e or "503" in e:
        return "gateway/api"
    if "timeout" in e or "timed out" in e:
        return "timeout"
    if "setup" in e or "install failed" in e or "agentsetup" in e:
        return "setup/install"
    if "oom" in e or "out of memory" in e:
        return "oom"
    return "other-error"


# ── per-run summary build ────────────────────────────────────────────────────────────────────────
#: ``.beagle/agents/<profile>.json`` — onboard files the version it pinned alongside the ref it
#: seeded. That mapping is the only place a run's *human* version name survives: run.json records
#: ``agent.source.ref`` (a SHA) and no version, because RunConfig flattens the canonical
#: ``harness.version`` away. Built once, tolerant of a missing/partial directory.
_MANIFEST_DIR = Path(__file__).resolve().parents[2] / ".beagle" / "agents"


@lru_cache(maxsize=1)
def _versions_by_ref() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in sorted(_MANIFEST_DIR.glob("*.json")) if _MANIFEST_DIR.is_dir() else []:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ref, ver = str(m.get("ref") or ""), str(m.get("version") or "")
        if ref and ver:
            out[ref] = ver
    return out


def _version_from_config(config_path: str | None) -> str:
    """``agent.harness.version`` from the canonical config the run was launched with.

    The most authoritative source: it is what the run DECLARED, e.g. ``20260826``. ``run.json``
    keeps the path but not the value, because RunConfig flattens ``harness`` away. ``""`` when
    the path is missing, moved, or unreadable — configs get reorganised, so this must not raise.
    """
    if not config_path:
        return ""
    try:
        doc = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return ""
    harness = ((doc.get("agent") or {}).get("harness") or {})
    return str(harness.get("version") or "")


def harness_version(agent: dict[str, Any], config_path: str | None = None) -> str:
    """``<version>_<ref6>`` for a run — the declared version AND the exact code behind it.

    Both halves earn their place. The version alone can't tell two runs apart once evolution
    starts, since every candidate shares the baseline's version; the ref alone is unreadable and
    loses the name the harness was onboarded under. Together they stay meaningful in both modes:
    ``20260826_53a76a`` for a baseline, and the same version with a different suffix for each
    candidate branch derived from it.

    Version resolution, best first:

    1. ``agent.harness.version`` in the canonical config the run recorded — what it declared.
    2. The onboarding manifest, keyed by ref, when that config has moved or been deleted.
    3. Neither: fall back to the bare short ref. Deliberately NOT parsed out of the run
       directory name, which only carries a version for generator-produced runs and would need
       every benchmark tag to split correctly — brittle, and wrong silently.
    """
    ref = str(((agent.get("source") or {}).get("ref")) or "")
    version = _version_from_config(config_path) or _versions_by_ref().get(ref, "")
    if version and ref:
        return f"{version}_{ref[:6]}"
    if version:
        return version
    return ref[:12] if ref else "?"


def _cfg_meta(cfg: dict[str, Any], config_path: str | None = None) -> tuple[str, str, str, str]:
    agent = cfg.get("agent") or {}
    harness = agent.get("name") or "?"
    model = (cfg.get("model") or {}).get("name") or (agent.get("model") or {}).get("name") or "?"
    effort = (agent.get("config") or {}).get("effort") or "—"
    return harness, harness_version(agent, config_path), model, effort


def _config_from_disk(run_dir: Path) -> dict[str, Any]:
    for b in run_dir.iterdir():
        cj = b / "config.json"
        if cj.exists() and not cj.is_dir():
            try:
                return json.loads(cj.read_text())
            except (OSError, ValueError):
                pass
    parts = run_dir.name.split("-")[0].split("_")
    if len(parts) >= 4:
        return {"agent": {"name": parts[0], "config": {"effort": parts[3]}}, "model": {"name": parts[2]}}
    return {}


def _trial_dirs(bench_dir: Path) -> list[Path]:
    return sorted(d for d in bench_dir.iterdir()
                  if d.is_dir() and ((d / "agent").is_dir() or (d / "result.json").exists()))


def _trial_row(trial_dir: Path, harness: str) -> dict[str, Any]:
    reward = err = None
    d: dict[str, Any] = {}
    try:
        d = json.loads((trial_dir / "result.json").read_text())
        reward = (d.get("reward")
                  if d.get("reward") is not None
                  else ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward"))
        md = (d.get("agent_result") or {}).get("metadata") or {}
        err = md.get("error") or d.get("error") or d.get("exception_info")
        if err is not None:
            err = str(err)
    except (OSError, ValueError):
        pass
    resolved = bool(reward is not None and reward >= 1.0)
    # Prefer the harness-written result.json "tokens" (the swe-bench / DockerHarness path records the
    # canonical, cache-split tokens there — and doesn't sync the raw agent stream to agent/); fall back
    # to parsing the raw stream (harbor/pier path, where result.json is harbor-native with no "tokens").
    rj_tokens = d.get("tokens")
    tokens = _norm_tokens(rj_tokens) if rj_tokens else trial_tokens(trial_dir / "agent", harness)
    return {
        "task_id": trial_dir.name,
        "reward": reward,
        "resolved": resolved,
        "error": (err[:300] if err else None),
        "error_type": classify_error(err, resolved),
        "tokens": tokens,
        "agent_seconds": agent_seconds(d),   # agent-execution phase only (excl. setup + verifier)
        "trajectory": str((trial_dir / "agent" / "trajectory.json").relative_to(trial_dir.parents[1])),
    }


def build_run_summary(run_dir: Path) -> dict[str, Any]:
    """Parse a run's trials into a compact summary and cache it to ``<run>/summary.json``."""
    rj = run_dir / "run.json"
    run = json.loads(rj.read_text()) if rj.exists() else {}
    cfg = run.get("config") or _config_from_disk(run_dir)
    harness, version, model, effort = _cfg_meta(cfg, run.get("config_path"))

    bench_meta = run.get("benchmarks") or {}
    bench_names = list(bench_meta) or [b.name for b in run_dir.iterdir()
                                       if b.is_dir() and (b / "config.json").exists()]
    benchmarks: dict[str, Any] = {}
    grand = _norm_tokens(None)
    for name in bench_names:
        bench_dir = run_dir / ((bench_meta.get(name) or {}).get("job_dir") or name)
        if not bench_dir.is_dir():
            continue
        trials = [_trial_row(td, harness) for td in _trial_dirs(bench_dir)]
        tok = _norm_tokens(None)
        for t in trials:
            tok = add_tokens(tok, t["tokens"])
        n_res = sum(1 for t in trials if t["resolved"])
        # trust run.json's grader score/counts when present; else derive from trials
        prev = bench_meta.get(name) or {}
        num_tasks = prev.get("num_tasks") if prev.get("num_tasks") is not None else len(trials)
        num_res = prev.get("num_resolved") if prev.get("num_resolved") is not None else n_res
        lats = [t["agent_seconds"] for t in trials if t.get("agent_seconds") is not None]
        benchmarks[name] = {
            "num_tasks": num_tasks, "num_resolved": num_res,
            "score": prev.get("score", ((num_res or 0) / (num_tasks or 1))),
            "median_latency_sec": (statistics.median(lats) if lats else None),
            "tokens": tok, "trials": trials,
        }
        grand = add_tokens(grand, tok)

    summary = {
        "_schema": _SCHEMA,
        "runname": run_dir.name, "harness": harness, "version": version,
        "model": model, "effort": effort,
        "benchmarks": benchmarks,
        "totals": {"tokens": grand,
                   "num_tasks": sum(b["num_tasks"] for b in benchmarks.values()),
                   "num_resolved": sum(b["num_resolved"] for b in benchmarks.values())},
        "in_progress": not rj.exists(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# ── discovery + loading ──────────────────────────────────────────────────────────────────────────
def discover_runs(results_dir: Path = RESULTS_DIR) -> list[Path]:
    out = []
    for p in sorted(results_dir.iterdir()):
        if not p.is_dir() or p.name in _SKIP_DIRS:
            continue
        if (p / "run.json").exists() or any(
                (b / "config.json").exists() for b in p.iterdir() if b.is_dir()):
            out.append(p)
    return out


def load_summaries(results_dir: Path = RESULTS_DIR, *, rebuild: bool = False) -> list[dict[str, Any]]:
    """Load each run's cached ``summary.json``, building it if missing or ``rebuild``."""
    summaries = []
    for run_dir in discover_runs(results_dir):
        sj = run_dir / "summary.json"
        if not rebuild and sj.exists():
            try:
                cached = json.loads(sj.read_text())
                if cached.get("_schema") == _SCHEMA:   # stale-schema caches fall through to rebuild
                    summaries.append(cached)
                    continue
            except (OSError, ValueError):
                pass
        try:
            summaries.append(build_run_summary(run_dir))
        except Exception as e:  # noqa: BLE001
            print(f"  ! {run_dir.name}: {e}")
    return summaries


# ── derived views (cheap, in-memory) ─────────────────────────────────────────────────────────────
def cost_of(tokens: dict[str, int], model: str, prices: dict[str, dict[str, float]]) -> float:
    p = prices.get(model) or prices.get("*") or {"input": 0, "cached": 0, "output": 0}
    fresh = max(0, tokens.get("prompt", 0) - tokens.get("cache_read", 0))
    return (fresh * p["input"] + tokens.get("cache_read", 0) * p["cached"]
            + tokens.get("completion", 0) * p["output"]) / 1_000_000


def overview_rows(summaries: list[dict[str, Any]], prices: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """One row per (run, benchmark) for the overview table."""
    rows = []
    for s in summaries:
        model = s["model"]
        for bname, b in s["benchmarks"].items():
            tok = b["tokens"]
            # median per-task cost — computed live from each trial's tokens (prices are editable), so it
            # can't be cached in summary.json. Over trials that produced tokens (a no-run trial is $0).
            trial_costs = [cost_of(t["tokens"], model, prices)
                           for t in b.get("trials", []) if t["tokens"].get("total")]
            median_cost = statistics.median(trial_costs) if trial_costs else None
            # SOLVED-ONLY variants, reported as SEPARATE columns rather than folded into the two
            # above. The all-task medians answer "what does attempting a task cost"; these answer
            # "what does solving one cost" — a failed trial can be cheap (died early) or expensive
            # (burned its whole budget getting nowhere), so mixing the two populations hides
            # exactly the comparison worth making between harnesses.
            solved = [t for t in b.get("trials", []) if t.get("resolved")]
            solved_lats = [t["agent_seconds"] for t in solved if t.get("agent_seconds") is not None]
            solved_costs = [cost_of(t["tokens"], model, prices)
                            for t in solved if t["tokens"].get("total")]
            rows.append({
                "Benchmark": bname, "Harness": s["harness"],
                # Right after Harness: which build of it produced these numbers.
                "Version": s.get("version") or "?",
                "Model": model, "Effort": s["effort"],
                "Resolved": b["num_resolved"], "Tasks": b["num_tasks"],
                "Score": round(b["score"], 4),
                "Total tokens": tok["total"], "Cached tokens": tok["cache_read"],
                "Cache %": round(100 * tok["cache_read"] / tok["prompt"], 1) if tok["prompt"] else 0.0,
                "Cost (est. $)": round(cost_of(tok, model, prices), 2),
                # median agent-execution seconds per task (excl. setup + verifier); None if untimed
                "Latency/task (s)": (round(b["median_latency_sec"]) if b.get("median_latency_sec")
                                     is not None else None),
                "Cost/task ($)": (round(median_cost, 2) if median_cost is not None else None),
                # Hidden in the UI unless the viewer opts in; always computed (it is cheap).
                "[solved]Latency/task (s)": (round(statistics.median(solved_lats))
                                            if solved_lats else None),
                "[solved]Cost/task ($)": (round(statistics.median(solved_costs), 2)
                                         if solved_costs else None),
                "Run": s["runname"], "In progress": s.get("in_progress", False),
            })
    return rows
