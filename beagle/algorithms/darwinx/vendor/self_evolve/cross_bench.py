"""Cross-benchmark fitness — the anti-overfit selection pressure.

A harness variant is only a *general* improvement if it Preserves+Extends on a
HELD-OUT benchmark it was **not** evolved against. Evolving + scoring + gating a
variant entirely on one benchmark (today: terminal-bench-v2) rewards variants
that memorize that benchmark's task families (e.g. a "chess" or "gcode" skill)
without transferring. This module makes generality a *gate*: after a candidate
passes the in-domain screen, it is re-scored on a small held-out benchmark
subset, and a candidate that regresses there is vetoed (never promoted).

Layered like the rest of the pipeline:
  * PURE core (`cross_bench_verdict`, `apply_cross_bench`) — unit-testable with
    no agent run.
  * Seam glue (`run_heldout_rates`) — reuses `codingbench_eval.run_subset_sampled`
    with a benchmark-swapped `CodingBenchEvalConfig` (via `dataclasses.replace`).

All behaviour is env-gated OFF by default so the existing (single-benchmark)
control arm is byte-for-byte unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


# ── env knobs (ATELIER_* convention; default OFF / conservative) ─────────

def gate_enabled() -> bool:
    return os.environ.get("ATELIER_CROSS_BENCH_GATE", "").strip() in ("1", "true", "True")


def heldout_benchmark() -> str:
    return os.environ.get("ATELIER_HELDOUT_BENCHMARK", "swe-bench-verified").strip()


def heldout_dataset() -> str | None:
    v = os.environ.get("ATELIER_HELDOUT_DATASET", "").strip()
    return v or None


def heldout_tasks() -> list[str]:
    """Held-out task subset: ``ATELIER_HELDOUT_TASKS`` as CSV, or ``@/path`` to a
    newline-separated file (``#`` comments allowed). Empty => gate is skipped."""
    raw = os.environ.get("ATELIER_HELDOUT_TASKS", "").strip()
    if not raw:
        return []
    if raw.startswith("@"):
        try:
            lines = Path(raw[1:]).expanduser().read_text().splitlines()
        except OSError:
            return []
        return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return [t.strip() for t in raw.split(",") if t.strip()]


def heldout_k() -> int:
    try:
        return max(1, int(os.environ.get("ATELIER_HELDOUT_K", "1")))
    except (TypeError, ValueError):
        return 1


def regression_margin() -> float:
    """A held-out task counts as regressed only if it drops by MORE than this
    (guards against sampling noise at small k). Default 0.0 = any drop."""
    try:
        return max(0.0, float(os.environ.get("ATELIER_CROSS_BENCH_MARGIN", "0.0")))
    except (TypeError, ValueError):
        return 0.0


def heldout_baseline() -> dict[str, float]:
    """Parent/root held-out pass-rates the candidate is judged against.

    Supplied out-of-band so the gate needs no parent re-checkout: compute it once
    on the campaign root commit via :func:`run_heldout_rates` and hand it in as
    ``ATELIER_HELDOUT_BASELINE`` — either ``@/path.json`` (a ``{task: rate}``
    mapping) or inline ``task=rate,task=rate``. Empty => gate skips (no false
    vetoes)."""
    raw = os.environ.get("ATELIER_HELDOUT_BASELINE", "").strip()
    if not raw:
        return {}
    if raw.startswith("@"):
        import json
        try:
            data = json.loads(Path(raw[1:]).expanduser().read_text())
        except (OSError, ValueError):
            return {}
        out: dict[str, float] = {}
        for k, v in (data or {}).items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out
    out = {}
    for pair in raw.split(","):
        if "=" in pair:
            k, _, v = pair.partition("=")
            try:
                out[k.strip()] = float(v)
            except (TypeError, ValueError):
                continue
    return out


# ── pure verdict core ───────────────────────────────────────────────────

@dataclass(frozen=True)
class CrossBenchVerdict:
    verdict: str            # "extend" | "preserve" | "regress" | "skip"
    preserved: bool
    extended: bool
    regressed_tasks: list[str]
    improved_tasks: list[str]
    parent_mean: float
    child_mean: float
    n_common: int
    detail: str


def cross_bench_verdict(
    parent_rates: dict[str, float],
    child_rates: dict[str, float],
    *,
    margin: float = 0.0,
) -> CrossBenchVerdict:
    """Preserve+Extend judgement on the held-out benchmark. Only tasks present in
    BOTH are compared."""
    common = sorted(set(parent_rates) & set(child_rates))
    if not common:
        return CrossBenchVerdict("skip", True, False, [], [], 0.0, 0.0, 0,
                                 "no held-out tasks common to parent and child")
    regressed = [t for t in common if child_rates[t] < parent_rates[t] - margin]
    improved = [t for t in common if child_rates[t] > parent_rates[t] + margin]
    pmean = sum(parent_rates[t] for t in common) / len(common)
    cmean = sum(child_rates[t] for t in common) / len(common)
    preserved = not regressed
    extended = bool(improved) and cmean >= pmean - 1e-9
    if regressed:
        verdict = "regress"
    elif extended:
        verdict = "extend"
    else:
        verdict = "preserve"
    detail = (f"held-out {heldout_benchmark()}: mean {pmean:.3f}->{cmean:.3f}, "
              f"+{len(improved)}/-{len(regressed)} of {len(common)} tasks")
    return CrossBenchVerdict(verdict, preserved, extended, regressed, improved,
                             pmean, cmean, len(common), detail)


_PROMOTE, _ARCHIVE, _REJECT = "promote", "archive", "reject"


def apply_cross_bench(base_verdict: str, cb: CrossBenchVerdict) -> tuple[str, str]:
    """Combine the in-domain gate verdict with the held-out result. Cross-bench
    can only *downgrade* — never turn a REJECT into a PROMOTE."""
    base = (base_verdict or "").strip().lower()
    if cb.verdict == "skip":
        return base, f"cross-bench skipped ({cb.detail})"
    if cb.verdict == "regress":
        if base == _PROMOTE:
            return _ARCHIVE, f"cross-bench regressed -> capped promote->archive; {cb.detail}"
        if base == _ARCHIVE:
            return _ARCHIVE, f"cross-bench regressed but kept as stepping stone; {cb.detail}"
        return _REJECT, f"cross-bench regressed and no in-domain promotion; {cb.detail}"
    return base, f"cross-bench {cb.verdict} confirmed generality; {cb.detail}"


# ── seam glue ────────────────────────────────────────────────────────────

def run_heldout_rates(
    base_cb_cfg,
    task_names: list[str],
    *,
    k_samples: int | None = None,
    config_path: Path | None = None,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
    tee_log_path: Path | None = None,
) -> dict[str, float]:
    """Score ``task_names`` on the held-out benchmark, returning task->pass-rate.
    Reuses ``codingbench_eval.run_subset_sampled`` with a benchmark-swapped cfg."""
    from . import codingbench_eval as cbe

    ho_cfg = replace(
        base_cb_cfg,
        benchmark_name=heldout_benchmark(),
        dataset=(heldout_dataset() or getattr(base_cb_cfg, "dataset")),
    )
    res = cbe.run_subset_sampled(
        cb_cfg=ho_cfg,
        task_names=task_names,
        k_samples=k_samples or heldout_k(),
        config_path=config_path,
        cwd=cwd,
        extra_env=extra_env,
        tee_log_path=tee_log_path,
    )
    return {t: rate for t, (rate, _n) in res.rates.items()}


__all__ = [
    "gate_enabled", "heldout_benchmark", "heldout_dataset", "heldout_tasks",
    "heldout_k", "regression_margin", "heldout_baseline",
    "CrossBenchVerdict", "cross_bench_verdict", "apply_cross_bench",
    "run_heldout_rates",
]
