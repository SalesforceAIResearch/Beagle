"""Variance-aware statistics for denoised self-evolution decisions.

Why this exists
---------------
The overnight ``atelier_evolve_tb21`` campaign accepted **zero** improvements:
inner-loop fitness (mini-eval) and the equivalence gate both ran at **k=1**, so
a single-sample "regression" or "improvement" is dominated by sampling noise.
A task that truly passes 4-of-5 times will "regress" on a k=1 probe ~20% of the
time -- which is exactly how the gate rejected every real fix.

This module turns repeated k-sample evaluations into variance-aware accept/
reject decisions used by the 3-stage cascade (mini-eval screen -> denoised probe
screen -> full k=5 confirm):

- ``pass_rate``           : avg@k from k binary trial outcomes
- ``wilson_interval``     : Wilson score CI for a pass-rate
- ``classify_task``       : per-task IMPROVED / REGRESSED / EQUIVALENT using a
                            noise-robust minimum-delta rule (+ optional CI guard)
- ``compare_task_sets``   : paired comparison over shared tasks -> the lists of
                            *real* improvements / regressions and the mean delta
- ``paired_bootstrap_delta`` : bootstrap CI on the mean per-task delta
- ``is_real_improvement`` : campaign-level "is this child really better than its
                            parent?" verdict (CI lower bound > 0, no net real
                            regressions)

Everything here is pure-stdlib and deterministic given a seed, so it is unit
testable without touching Harbor / the cluster. Run ``python -m
self_evolve.eval_stats`` for the built-in self-checks.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class TaskVerdict(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    EQUIVALENT = "equivalent"


# Default decision knobs. ``MIN_RATE_DELTA`` is the smallest pass-rate change we
# treat as real rather than noise. At k=5 a single-trial flip is 1/5 = 0.20, so
# a 0.40 threshold (>= 2/5 swing) ignores one-off flips while still catching a
# genuine 5/5 -> <=3/5 regression or a 0/5 -> >=2/5 fix.
DEFAULT_MIN_RATE_DELTA = 0.40
# Confidence for the Wilson interval / bootstrap CI (z=1.96 -> ~95%).
DEFAULT_Z = 1.96
DEFAULT_BOOTSTRAP_N = 2000
DEFAULT_BOOTSTRAP_SEED = 1234


def pass_rate(outcomes: Sequence[float]) -> float:
    """avg@k: mean of binary (>=1.0 treated as pass) trial outcomes."""
    if not outcomes:
        return 0.0
    return sum(1.0 if o >= 1.0 else 0.0 for o in outcomes) / len(outcomes)


def wilson_interval(n_pass: int, n: int, z: float = DEFAULT_Z) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial pass-rate.

    More reliable than the normal approximation at the small k (3-5) we run per
    task. Returns ``(lo, hi)`` clamped to [0, 1]; an empty sample is ``(0, 1)``.
    """
    if n <= 0:
        return (0.0, 1.0)
    phat = n_pass / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class TaskComparison:
    task: str
    parent_rate: float
    child_rate: float
    n_parent: int
    n_child: int
    verdict: TaskVerdict
    delta: float


def classify_task(
    *,
    parent_rate: float,
    child_rate: float,
    n_parent: int,
    n_child: int,
    min_delta: float = DEFAULT_MIN_RATE_DELTA,
    use_ci_guard: bool = True,
    z: float = DEFAULT_Z,
) -> TaskVerdict:
    """Classify one task's child-vs-parent change, robust to k-sample noise.

    A change counts only if the pass-rate moved by at least ``min_delta``. When
    ``use_ci_guard`` is set we additionally require the Wilson intervals to be
    consistent with the direction (the child's CI must clear the parent's point
    estimate), which suppresses borderline calls when k is tiny.
    """
    delta = child_rate - parent_rate
    if abs(delta) < min_delta:
        return TaskVerdict.EQUIVALENT

    if not use_ci_guard:
        return TaskVerdict.IMPROVED if delta > 0 else TaskVerdict.REGRESSED

    child_lo, child_hi = wilson_interval(round(child_rate * n_child), n_child, z)
    if delta > 0:
        # Real improvement: child's lower CI is above the parent's point rate.
        return TaskVerdict.IMPROVED if child_lo > parent_rate else TaskVerdict.EQUIVALENT
    # Real regression: child's upper CI is below the parent's point rate.
    return TaskVerdict.REGRESSED if child_hi < parent_rate else TaskVerdict.EQUIVALENT


@dataclass
class SetComparison:
    """Paired comparison of a child vs its parent over shared tasks."""
    improved: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)
    equivalent: list[str] = field(default_factory=list)
    mean_delta: float = 0.0
    ci_lo: float = 0.0
    ci_hi: float = 0.0
    n_tasks: int = 0
    per_task: list[TaskComparison] = field(default_factory=list)

    @property
    def net_real_gain(self) -> int:
        return len(self.improved) - len(self.regressed)


def paired_bootstrap_delta(
    deltas: Sequence[float],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    z: float = DEFAULT_Z,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap CI on the mean of per-task deltas.

    Returns ``(mean_delta, ci_lo, ci_hi)``. ``z`` maps to the CI width via the
    normal quantile (1.96 -> 95%); we use percentile bootstrap, so ``z`` only
    sets the tail fraction.
    """
    if not deltas:
        return (0.0, 0.0, 0.0)
    mean = sum(deltas) / len(deltas)
    if len(deltas) == 1:
        return (mean, mean, mean)
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    # two-sided tail fraction from z (normal approx): alpha = 2*(1-Phi(z))
    alpha = 2.0 * (1.0 - _phi(z))
    lo_idx = max(0, int((alpha / 2.0) * n_boot))
    hi_idx = min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot))
    return (mean, means[lo_idx], means[hi_idx])


def compare_task_sets(
    parent_rates: Mapping[str, tuple[float, int]],
    child_rates: Mapping[str, tuple[float, int]],
    *,
    min_delta: float = DEFAULT_MIN_RATE_DELTA,
    use_ci_guard: bool = True,
    z: float = DEFAULT_Z,
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> SetComparison:
    """Compare child vs parent over the tasks they share.

    ``*_rates`` map task -> ``(pass_rate, n_samples)``. Only tasks present in
    both are compared (the gate caches the parent's rates once; the probe
    samples the child). Returns the real improvements/regressions and a
    bootstrap CI on the mean per-task delta.
    """
    shared = [t for t in parent_rates if t in child_rates]
    out = SetComparison(n_tasks=len(shared))
    deltas: list[float] = []
    for t in shared:
        p_rate, n_p = parent_rates[t]
        c_rate, n_c = child_rates[t]
        verdict = classify_task(
            parent_rate=p_rate, child_rate=c_rate, n_parent=n_p, n_child=n_c,
            min_delta=min_delta, use_ci_guard=use_ci_guard, z=z,
        )
        tc = TaskComparison(t, p_rate, c_rate, n_p, n_c, verdict, c_rate - p_rate)
        out.per_task.append(tc)
        deltas.append(c_rate - p_rate)
        if verdict is TaskVerdict.IMPROVED:
            out.improved.append(t)
        elif verdict is TaskVerdict.REGRESSED:
            out.regressed.append(t)
        else:
            out.equivalent.append(t)
    out.mean_delta, out.ci_lo, out.ci_hi = paired_bootstrap_delta(
        deltas, n_boot=n_boot, z=z, seed=seed,
    )
    return out


def is_real_improvement(
    cmp: SetComparison,
    *,
    require_ci_positive: bool = True,
    allow_regressions: int = 0,
) -> bool:
    """Campaign-level accept rule: a child is a *real* improvement iff the mean
    per-task delta's CI excludes 0 on the positive side (when
    ``require_ci_positive``) and it introduces no more than ``allow_regressions``
    real regressions. This replaces the noise-gamed strict ``child > parent``.
    """
    if len(cmp.regressed) > allow_regressions:
        return False
    if require_ci_positive:
        return cmp.ci_lo > 0.0
    return cmp.mean_delta > 0.0 and cmp.net_real_gain > 0


def is_real_regression(
    cmp: SetComparison,
    *,
    max_regressions: int = 0,
) -> bool:
    """Gate veto rule: reject iff the child has more than ``max_regressions``
    *statistically real* regressions (noise-robust), not "any single probe
    failed once"."""
    return len(cmp.regressed) > max_regressions


def _phi(z: float) -> float:
    """Standard-normal CDF via erf (stdlib)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _selftest() -> None:
    # 1) noise-robust: parent 5/5, child 4/5 -> EQUIVALENT (a single flip).
    assert classify_task(parent_rate=1.0, child_rate=0.8, n_parent=5, n_child=5) \
        is TaskVerdict.EQUIVALENT
    # 2) real regression: parent 5/5, child 2/5 -> REGRESSED.
    assert classify_task(parent_rate=1.0, child_rate=0.4, n_parent=5, n_child=5) \
        is TaskVerdict.REGRESSED
    # 3) real fix: parent 0/5, child 4/5 -> IMPROVED.
    assert classify_task(parent_rate=0.0, child_rate=0.8, n_parent=5, n_child=5) \
        is TaskVerdict.IMPROVED
    # 4) set comparison: one real fix, one noise flip -> net +1, no regressions.
    parent = {"fix_me": (0.0, 5), "stable": (1.0, 5)}
    child = {"fix_me": (0.8, 5), "stable": (0.8, 5)}
    cmp = compare_task_sets(parent, child)
    assert cmp.improved == ["fix_me"], cmp.improved
    assert cmp.regressed == [], cmp.regressed
    assert is_real_improvement(cmp, require_ci_positive=False)
    # 5) a pure-noise child is NOT a real improvement.
    parent2 = {"a": (1.0, 5), "b": (1.0, 5), "c": (0.0, 5)}
    child2 = {"a": (0.8, 5), "b": (1.0, 5), "c": (0.2, 5)}
    cmp2 = compare_task_sets(parent2, child2)
    assert cmp2.improved == [] and cmp2.regressed == [], (cmp2.improved, cmp2.regressed)
    assert not is_real_improvement(cmp2)
    print("eval_stats self-test OK")


if __name__ == "__main__":
    _selftest()
