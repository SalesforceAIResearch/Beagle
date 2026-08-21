"""The ``run.json`` record — the run's canonical, crash-safe summary on disk.

Run-level record shape + atomic tempfile→replace write. The native per-benchmark artifact trees (harbor's
``<job>/<trial>/…``) nest *inside* the run dir and are the crash-safe per-task record;
``run.json`` is the summary the grader/analysis read.

The Runner (:mod:`beagle.rollout.runner`) computes ``metrics`` + ``totals`` and calls
:func:`write_run_json`; these helpers stay pure so they're trivially unit-tested.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from beagle.types import TaskResult

if TYPE_CHECKING:
    from beagle.config import RunConfig


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as JSON so a partial file never appears (tempfile + ``os.replace``)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def capture_environment() -> dict[str, Any]:
    """Provenance: python + platform + git commit/dirty (best-effort, never raises)."""
    def _git(*args: str) -> str:
        try:
            r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:  # noqa: BLE001 - provenance is best-effort
            return ""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD") or None,
        "dirty": bool(_git("status", "--porcelain")),
    }


def per_task_row(r: TaskResult, *, benchmark: str) -> dict[str, Any]:
    """One TaskResult → a compact in-memory row the per-benchmark aggregates derive from.

    NOT written to disk — ``run.json`` is a thin, benchmark-keyed *derived* summary, and
    the per-task source of truth is each harness's native tree (harbor's
    ``<benchmark>/<trial>/result.json``, read back on resume via ``harness.completed``).
    ``trial_dir`` lets the Runner recover the benchmark's ``job_dir`` from where the harness
    actually wrote."""
    return {
        "task_id": r.task_id,
        "benchmark": benchmark,
        "resolved": r.resolved,
        "reward": r.reward,
        "tokens": dict(r.tokens),
        "error": r.error,
        "trial_dir": str(r.artifact_dir) if r.artifact_dir else None,
    }


#: Additive token keys carried through to the run/benchmark totals. ``prompt``/``completion`` are the
#: legacy billable input/output; ``input_uncached``/``cache_read``/``cache_write`` are the cache split
#: a downstream cost estimate needs (invariant: prompt = input_uncached + cache_read + cache_write).
_TOKEN_KEYS = ("prompt", "completion", "input_uncached", "cache_read", "cache_write")


def _sum_tokens(rows: list[dict[str, Any]]) -> dict[str, int]:
    out = {k: sum(int((r.get("tokens") or {}).get(k, 0)) for r in rows) for k in _TOKEN_KEYS}
    out["total"] = out["prompt"] + out["completion"]
    return out


def benchmark_summary(
    rows: list[dict[str, Any]], *, score: float, job_dir: str | None, wall_time_sec: float | None
) -> dict[str, Any]:
    """Per-benchmark derived aggregate — score + counts + token sums, pointing at the
    benchmark's native artifact subtree (``job_dir``, relative to the run dir) where the
    per-task detail lives. ``score`` comes from the benchmark's OWN grader (reward
    semantics differ per benchmark — it is NOT comparable across benchmarks)."""
    return {
        "job_dir": job_dir,
        "score": score,
        "num_tasks": len(rows),
        "num_resolved": sum(1 for r in rows if r.get("resolved")),
        "num_errored": sum(1 for r in rows if r.get("error") is not None),
        "tokens": _sum_tokens(rows),
        "wall_time_sec": wall_time_sec,
    }


def compute_totals(
    rows: list[dict[str, Any]], *, num_benchmarks: int, wall_time_sec: float | None = None
) -> dict[str, Any]:
    """Run-wide tallies — ONLY quantities that are additive across benchmarks (counts,
    tokens, wall-time). Deliberately **no cross-benchmark ``score``**: reward semantics
    differ per benchmark, so averaging them is meaningless (per-benchmark ``score`` lives
    under ``benchmarks``). ``resolved_rate`` is a raw count ratio (resolved / tasks), a
    defined number for any mix — not a fair score comparison."""
    n = len(rows)
    resolved = sum(1 for r in rows if r.get("resolved"))
    return {
        "num_benchmarks": num_benchmarks,
        "num_tasks": n,
        "num_attempted": sum(1 for r in rows if r.get("error") is None),
        "num_resolved": resolved,
        "num_errored": sum(1 for r in rows if r.get("error") is not None),
        "resolved_rate": (resolved / n) if n else 0.0,
        "tokens": _sum_tokens(rows),
        "wall_time_sec": wall_time_sec,
    }


def _compact(obj: Any) -> Any:
    """Recursively drop empty/absent fields (None, "", {}, []) so the embedded config
    reads clean — e.g. monet's routing lives in ``agent.config``, so ``model.provider``
    is empty and simply disappears rather than showing ``"provider": ""``."""
    if isinstance(obj, dict):
        out = {k: _compact(v) for k, v in obj.items()}
        return {k: v for k, v in out.items() if v not in (None, "", {}, [])}
    if isinstance(obj, list):
        return [_compact(v) for v in obj]
    return obj


def compact_config(config: RunConfig) -> dict[str, Any]:
    """The resolved RunConfig as a compact dict — a self-contained, reproducible record
    that replaces the old duplicated model/agent/benchmark blocks. ``exclude_defaults``
    drops fields left at their default (so ``tag: "main"``, ``num_samples: 1``, unset
    harbor image-coord knobs, etc. don't clutter it — only what the run actually specifies
    shows); ``_compact`` then prunes any remaining empties. The integrity anchor is
    ``config_hash`` (over the FULL config), so this lossy-of-defaults view is safe."""
    return _compact(config.model_dump(mode="json", exclude_defaults=True))


def assemble_run_record(
    *,
    run_id: str,
    config: RunConfig,
    benchmarks: dict[str, Any],
    totals: dict[str, Any],
    config_hash: str,
    config_path: str | Path | None = None,
    campaign_id: str | None = None,
    environment: dict[str, Any] | None = None,
    timestamp_start: str | None = None,
    timestamp_end: str | None = None,
    config_hash_drift: str | None = None,
) -> dict[str, Any]:
    """Build the run-level record: the *thin, benchmark-keyed* summary.

    Shape: identity + the embedded resolved ``config`` (+ ``config_path``/``config_hash``
    for provenance/integrity) + ``environment`` + per-benchmark aggregates (``benchmarks``)
    + additive ``totals``. No per-task array (per-task is recoverable from the harness's
    native trees, read via ``harness.completed``); no duplicated model/agent/benchmark
    blocks (folded into ``config``)."""
    return {
        "run_id": run_id,
        "campaign_id": campaign_id,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "config_path": str(config_path) if config_path else None,
        "config_hash": config_hash,
        # Set only when --force-resume overrode a drift guard: the PRIOR run.json's hash the
        # resumed tasks were produced under, kept for auditability (the run now mixes two configs).
        **({"config_hash_drift": config_hash_drift} if config_hash_drift else {}),
        "config": compact_config(config),
        "environment": environment or {},
        "benchmarks": benchmarks,
        "totals": totals,
    }


def write_run_json(run_dir: Path, record: dict[str, Any]) -> Path:
    """Atomically write ``<run_dir>/run.json`` and return its path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    _atomic_write_json(path, record)
    return path


__all__ = [
    "capture_environment", "per_task_row", "benchmark_summary", "compute_totals",
    "compact_config", "assemble_run_record", "write_run_json",
]
