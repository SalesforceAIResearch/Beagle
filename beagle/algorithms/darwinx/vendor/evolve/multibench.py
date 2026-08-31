"""Multi-benchmark fitness — scoring a candidate across a mixture, not one benchmark.

Single-benchmark evolution optimizes the benchmark, not the agent. `cross_bench.py` already
guards against that with a held-out *benchmark* veto, but it assumes one in-domain benchmark
and one held-out one. Evolving on a mixture (DeepSWE + SWE-bench-Pro + SWE-bench-Verified at
once) needs two things that module does not provide:

**Comparable scores.** Averaging raw pass-rates across benchmarks silently weights whichever
benchmark has the most headroom. If Verified sits at 0.55 and DeepSWE at 0.12, a candidate
can raise the mean by moving only Verified, and the mean cannot tell that apart from a
uniform gain. Worse, the benchmarks differ in how noisy they are, so the same +2 points is
strong evidence on one and nothing on another. So a benchmark's contribution is its gain over
its own baseline measured in its own replicate standard deviations -- a z-score. That makes
"one benchmark moved" and "all three moved" numerically different in the right direction.

**No trading one benchmark for another.** A weighted mean is happy to accept +3 sd on DeepSWE
against -2 sd on Verified, which is precisely the overfitting we are trying to prevent, just
laundered through a mixture. The floor is therefore a separate hard gate rather than a term
in the fitness: any benchmark that drops by more than its noise tolerance vetoes the
candidate no matter how good the aggregate looks.

The floor alone is not enough, though, and the reason is easy to miss: it only forbids going
*down*. Under a plain mean, a candidate that leaves two benchmarks untouched and moves the
third by 5 sd (score 1.67) beats one that moves all three by 1-2 sd (score 1.33), and neither
regresses, so the floor is silent. The mixture would then select for exactly the
single-benchmark specialist it was built to avoid. So gains are capped at
``DEFAULT_GAIN_CAP_SD`` before averaging: past a couple of sigma, further gain on one
benchmark stops buying score, and the only way to keep improving the aggregate is to move a
different benchmark. The cap is deliberately one-sided -- losses are averaged in at full
weight, because an aggregate that softened its own bad news would be worthless.

Layered like the rest of the pipeline: everything here is pure and unit-testable, with no
agent run and no cluster. The glue that actually scores each benchmark lives with the eval
seam; this module only decides what the numbers mean.

Env-gated OFF by default, so the existing single-benchmark arms are unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

#: Divisor floor for z-scores. A benchmark whose measured replicate sd is ~0 would otherwise
#: turn any trivial difference into an unbounded score. 0.01 = one point of pass-rate.
DEFAULT_SD_FLOOR = 0.01

#: How far a benchmark may fall before the floor vetoes, in units of its own sd. 1.0 means
#: "a one-sigma drop is tolerated"; noise at small k routinely moves a rate that far.
DEFAULT_TOL_SD = 1.0

#: Gains above this many sd stop counting toward the aggregate. Makes the fitness concave so
#: breadth beats depth; see the module docstring.
DEFAULT_GAIN_CAP_SD = 2.0

#: Tasks sampled per benchmark when the *gate* scores a candidate.
#:
#: The baselines are measured on the full corpus -- 500 Verified, 731 Pro, 113 DeepSWE -- and the
#: spec carries those full lists, because that is what the baseline and sd describe. The gate
#: cannot use them: 1344 trials per gated candidate, at minutes each, is not an evaluation
#: budget, it is the whole campaign spent on one node. So the gate scores a bounded, *fixed*
#: sample of each benchmark.
#:
#: Fixed, not fresh per candidate. Parent and child must be compared on the same tasks or the
#: difference between them picks up the variance between two task samples, which on a subset of
#: this size swamps the effect being measured. A seeded sample gives that for free and stays
#: reproducible across processes.
DEFAULT_GATE_TASKS = 80

#: Tasks per benchmark for the *screen*, the cheap first stage of the floor.
#:
#: 80 per benchmark is affordable per campaign but not per candidate: three benchmarks at 80, at
#: DeepSWE's ~24 minutes a trial, is hours of evaluation for every promotion, and at a hundred
#: nodes the floor would cost more than the evolution it guards. So the floor runs in two stages.
#: The screen is small and its tolerance is correspondingly wide -- at 25 tasks and p=0.8 it only
#: fires on a drop of ~11 points -- so a healthy candidate clears it and costs nothing more. Only
#: a candidate that looks like it regressed pays for the full sample, and only on the benchmarks
#: that looked bad.
#:
#: The screen sample is a *prefix* of the full sample (see ``gate_sample``), so escalating adds
#: tasks instead of discarding the trials already run.
DEFAULT_GATE_SCREEN_TASKS = 25


# ── env knobs (DARWINX_GATE_* convention; default OFF) ────────────────────────

def gate_enabled() -> bool:
    """Whether mixture scoring is active. Off => callers keep single-benchmark behaviour."""
    return os.environ.get("DARWINX_GATE_MIXTURE_GATE", "").strip() in ("1", "true", "True")


def tolerance_sd() -> float:
    try:
        return max(0.0, float(os.environ.get("DARWINX_GATE_MIXTURE_TOL_SD", DEFAULT_TOL_SD)))
    except (TypeError, ValueError):
        return DEFAULT_TOL_SD


def gain_cap_sd() -> float:
    """Where per-benchmark gains stop counting. ``0`` or negative disables the cap."""
    try:
        return float(os.environ.get("DARWINX_GATE_MIXTURE_GAIN_CAP_SD", DEFAULT_GAIN_CAP_SD))
    except (TypeError, ValueError):
        return DEFAULT_GAIN_CAP_SD


def min_abs_drop() -> float:
    """Absolute drop always tolerated, regardless of sd. Guards benchmarks whose measured sd
    came out implausibly small on few replicates."""
    try:
        return max(0.0, float(os.environ.get("DARWINX_GATE_MIXTURE_MIN_ABS_DROP", "0.0")))
    except (TypeError, ValueError):
        return 0.0


def _bench_env_suffix(benchmark: str) -> str:
    """``harbor-swebench-verified`` -> ``HARBOR_SWEBENCH_VERIFIED``."""
    return "".join(c if c.isalnum() else "_" for c in benchmark).upper()


def _sized_knob(base: str, default: int, benchmark: str | None) -> int:
    """A sample-size knob that can be set globally or for one benchmark.

    WHY PER-BENCHMARK SIZES EXIST
      A uniform sample charges the same for measurements that do not cost the same. A SWE-V trial
      is about five minutes; a Deep-SWE trial is about twenty-four, because it builds the repo
      under pier's phased network. At 25 tasks each that is ~12 minutes of wall clock for SWE-V
      and ~100 for Deep-SWE at the parallelism we have -- so a uniform screen spends most of the
      floor's budget on its slowest member, and over a hundred nodes the floor costs more than
      the evolution it guards.

      Unequal samples are sound here because the tolerance is computed from ``n_used`` rather
      than assumed: a smaller sample widens its own tolerance (see ``comparison_sd``), so a
      cheap benchmark buys precision and an expensive one buys coverage, and neither can veto on
      noise it does not have the power to see. What would be unsound is unequal samples with a
      shared tolerance.
    """
    for key in ((f"{base}_{_bench_env_suffix(benchmark)}",) if benchmark else ()) + (base,):
        raw = os.environ.get(key, "").strip()
        if raw:
            try:
                return int(raw)
            except (TypeError, ValueError):
                continue
    return default


def gate_tasks_n(benchmark: str | None = None) -> int:
    """Tasks sampled for the gate (``DARWINX_GATE_MIXTURE_GATE_TASKS``, or
    ``DARWINX_GATE_MIXTURE_GATE_TASKS_<BENCH>`` for one benchmark).

    ``0`` or negative means "use the whole list", which is only sane for a small benchmark.
    """
    return _sized_knob("DARWINX_GATE_MIXTURE_GATE_TASKS", DEFAULT_GATE_TASKS, benchmark)


def gate_screen_n(benchmark: str | None = None) -> int:
    """Tasks for the floor's cheap first stage (``DARWINX_GATE_MIXTURE_GATE_SCREEN_TASKS``, or
    ``DARWINX_GATE_MIXTURE_GATE_SCREEN_TASKS_<BENCH>``). ``0`` disables the screen: the floor then
    goes straight to the full sample, which is correct but expensive."""
    return _sized_knob("DARWINX_GATE_MIXTURE_GATE_SCREEN_TASKS", DEFAULT_GATE_SCREEN_TASKS, benchmark)


def gate_seed() -> int:
    """Seed for the gate's task sample (``DARWINX_GATE_MIXTURE_GATE_SEED``).

    Fixed across the campaign on purpose, so every candidate is scored on the same tasks and a
    parent-child difference is a difference in the agent rather than in the sample.
    """
    try:
        return int(os.environ.get("DARWINX_GATE_MIXTURE_GATE_SEED", "").strip() or 0)
    except (TypeError, ValueError):
        return 0


# ── the mixture spec ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class BenchComponent:
    """One benchmark's role in the mixture.

    ``baseline`` and ``sd`` are the root harness's measured pass-rate and its replicate
    standard deviation on this benchmark -- both come from the baseline campaign, not from
    anything the evolution computes, so a candidate cannot move its own yardstick.
    """

    name: str
    weight: float = 1.0
    baseline: float = 0.0
    sd: float = 0.0
    dataset: str | None = None
    tasks: list[str] = field(default_factory=list)
    #: How this benchmark is evaluated: ``split``, ``namespace``, ``tag``, ``registry``,
    #: ``options``. Written into the spec by the calibrator, copied from the benchmark's own
    #: baseline config.
    #:
    #: These cannot be inherited from the campaign's primary benchmark, which is what a plain
    #: ``replace(base_cfg, benchmark_name=...)`` would do. The members are not variations on one
    #: benchmark: SWE-V is a local harbor task directory whose images are named
    #: ``<registry>/swebench-verified:main-<task>``, while Pro and Lite are HuggingFace datasets
    #: on ``split: test`` whose loaders resolve their own images and are given no namespace at
    #: all. Inheriting SWE-V's three fields makes Pro look for SWE-V's images under SWE-V's
    #: namespace, which resolves to nothing -- and a benchmark that measures nothing is dropped
    #: from the floor rather than reported, so the campaign would quietly gate on one benchmark
    #: while its logs named three.
    eval: dict = field(default_factory=dict)

    def gate_order(self, seed: int | None = None) -> list[str]:
        """``tasks`` in a fixed pseudo-random order.

        Every gate sample is a prefix of this one ordering, which is what makes the floor's
        two stages compose: the screen's 25 tasks are the first 25 of the full 80, so escalating
        runs 55 more trials rather than 80 fresh ones.

        Deterministic in (benchmark name, seed), so every process and every candidate in a
        campaign draws the same tasks. The name is mixed in so two benchmarks do not both take
        their alphabetically-first tasks and correlate whatever that selects for.
        """
        import random

        pool = sorted(self.tasks)
        random.Random(f"{self.name}:{gate_seed() if seed is None else seed}").shuffle(pool)
        return pool

    def gate_sample(self, n: int | None = None, seed: int | None = None) -> list[str]:
        """The fixed subset of ``tasks`` the gate scores at full size.

        Sized per benchmark, because a Deep-SWE trial costs about five SWE-V trials and a
        uniform sample would spend the floor's budget on its slowest member.
        """
        want = gate_tasks_n(self.name) if n is None else n
        order = self.gate_order(seed)
        if want <= 0 or want >= len(order):
            return order
        return order[:want]

    def gate_screen_sample(self, seed: int | None = None) -> list[str]:
        """The floor's cheap first stage: a prefix of :meth:`gate_sample`."""
        n = gate_screen_n(self.name)
        full = self.gate_sample(seed=seed)
        return full if n <= 0 or n >= len(full) else full[:n]

    def comparison_sd(self, n_used: int) -> float:
        """Noise on a parent-child *difference* measured over ``n_used`` tasks.

        ``sd`` alone is the wrong tolerance for the gate, and tightly so. It describes the
        baseline's full-corpus rate -- 0.018 on 500 Verified tasks -- while the gate measures the
        candidate on 80. The sampling error there is sqrt(p(1-p)/80), about 0.045, more than
        twice as large. Using ``sd`` would make the floor veto healthy candidates on noise
        several times per generation, which looks exactly like a working safety gate and is the
        reason to write this down rather than discover it from a campaign that promoted nothing.

        Both sides carry that error, hence the factor of two under the root.
        """
        p = min(max(float(self.baseline), 0.0), 1.0)
        if n_used <= 0:
            return float(self.sd)
        se = (2.0 * p * (1.0 - p) / n_used) ** 0.5
        return (float(self.sd) ** 2 + se ** 2) ** 0.5


@dataclass(frozen=True)
class MixtureSpec:
    components: dict[str, BenchComponent] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return sorted(self.components)

    def component(self, name: str) -> BenchComponent | None:
        return self.components.get(name)


def load_mixture_spec(raw: str | None = None) -> MixtureSpec:
    """Parse ``DARWINX_GATE_MIXTURE_SPEC`` -- inline JSON or ``@/path.json``.

    Shape::

        {"benchmarks": {"swe-bench-verified": {"weight": 1.0, "baseline": 0.55,
                                                "sd": 0.028, "tasks": ["t1", "t2"]}, ...}}

    An unparseable or empty spec yields an empty mixture, which every consumer treats as
    "skip" -- a malformed spec must not silently become a veto or a fabricated score.
    """
    if raw is None:
        raw = os.environ.get("DARWINX_GATE_MIXTURE_SPEC", "")
    raw = (raw or "").strip()
    if not raw:
        return MixtureSpec()
    if raw.startswith("@"):
        try:
            raw = Path(raw[1:]).expanduser().read_text()
        except OSError:
            return MixtureSpec()
    try:
        data = json.loads(raw)
    except ValueError:
        return MixtureSpec()
    benchmarks = (data or {}).get("benchmarks") if isinstance(data, dict) else None
    if not isinstance(benchmarks, dict):
        return MixtureSpec()

    components: dict[str, BenchComponent] = {}
    for name, blob in benchmarks.items():
        if not isinstance(blob, dict):
            continue
        try:
            components[str(name)] = BenchComponent(
                name=str(name),
                weight=float(blob.get("weight", 1.0)),
                baseline=float(blob.get("baseline", 0.0)),
                sd=float(blob.get("sd", 0.0)),
                dataset=blob.get("dataset") or None,
                tasks=[str(t) for t in (blob.get("tasks") or [])],
                eval=dict(blob.get("eval") or {}),
            )
        except (TypeError, ValueError):
            continue
    return MixtureSpec(components)


# ── fitness ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MixtureFitness:
    score: float                        # weighted mean of capped per-benchmark z-scores
    per_bench_z: dict[str, float]       # uncapped, so the raw picture stays visible
    per_bench_rate: dict[str, float]
    scored: list[str]                   # benchmarks that had a rate to score
    missing: list[str]                  # in the spec but absent from the rates
    detail: str


def normalized_gain(rate: float, comp: BenchComponent, *, sd_floor: float = DEFAULT_SD_FLOOR) -> float:
    """This benchmark's gain over its baseline, in its own standard deviations."""
    return (rate - comp.baseline) / max(comp.sd, sd_floor)


def mixture_fitness(
    rates: dict[str, float],
    spec: MixtureSpec,
    *,
    sd_floor: float = DEFAULT_SD_FLOOR,
    gain_cap: float = DEFAULT_GAIN_CAP_SD,
) -> MixtureFitness:
    """Aggregate per-benchmark pass-rates into one baseline-normalized score.

    Gains are capped at ``gain_cap`` sd before averaging so that breadth outranks depth (see
    the module docstring); losses are not capped. Pass ``gain_cap<=0`` for a plain mean.

    Only benchmarks present in both the spec and ``rates`` contribute. A benchmark that
    failed to produce a rate is reported in ``missing`` rather than being treated as a zero,
    which would read as a catastrophic regression and veto a candidate over an
    infrastructure failure.
    """
    per_z: dict[str, float] = {}
    per_rate: dict[str, float] = {}
    total_w = 0.0
    acc = 0.0
    for name in spec.names:
        comp = spec.components[name]
        if name not in rates:
            continue
        rate = float(rates[name])
        z = normalized_gain(rate, comp, sd_floor=sd_floor)
        per_z[name] = z
        per_rate[name] = rate
        acc += comp.weight * (min(z, gain_cap) if gain_cap > 0 else z)
        total_w += comp.weight

    scored = sorted(per_z)
    missing = [n for n in spec.names if n not in per_z]
    score = (acc / total_w) if total_w > 0 else 0.0
    if not scored:
        detail = "mixture: no benchmark produced a rate"
    else:
        parts = ", ".join(
            f"{n} {per_rate[n]:.3f} (z={per_z[n]:+.2f}"
            + (f", capped at {gain_cap:+.1f}" if gain_cap > 0 and per_z[n] > gain_cap else "")
            + ")"
            for n in scored
        )
        detail = f"mixture score {score:+.3f} over {len(scored)} benchmark(s): {parts}"
        if missing:
            detail += f"; missing {', '.join(missing)}"
    return MixtureFitness(score, per_z, per_rate, scored, missing, detail)


# ── the per-benchmark regression floor ───────────────────────────────────

@dataclass(frozen=True)
class FloorVerdict:
    passed: bool
    regressed: list[str]
    drops: dict[str, float]             # positive = fell by this much
    tolerances: dict[str, float]
    detail: str


def regression_floor_verdict(
    parent_rates: dict[str, float],
    child_rates: dict[str, float],
    spec: MixtureSpec,
    *,
    tol_sd: float = DEFAULT_TOL_SD,
    min_abs: float = 0.0,
    n_used: dict[str, int] | None = None,
) -> FloorVerdict:
    """Veto if any single benchmark fell further than its own noise tolerance.

    Deliberately independent of :func:`mixture_fitness`: the point is that no amount of
    aggregate gain buys a regression on one member of the mixture. Only benchmarks with both
    a parent and a child rate are compared -- a missing measurement is not evidence of a drop.
    """
    drops: dict[str, float] = {}
    tolerances: dict[str, float] = {}
    regressed: list[str] = []
    for name in spec.names:
        if name not in parent_rates or name not in child_rates:
            continue
        comp = spec.components[name]
        drop = float(parent_rates[name]) - float(child_rates[name])
        # Tolerance is the noise on the *difference at the size actually measured*, not the
        # baseline's full-corpus sd -- see BenchComponent.comparison_sd. Callers that measured a
        # screen instead of the full sample pass the real size in ``n_used``; getting this wrong
        # is what makes a two-stage floor veto on noise at the cheap stage.
        n = (n_used or {}).get(name) or len(comp.gate_sample())
        tol = max(min_abs, tol_sd * comp.comparison_sd(n))
        drops[name] = drop
        tolerances[name] = tol
        if drop > tol + 1e-9:
            regressed.append(name)

    if not drops:
        return FloorVerdict(True, [], {}, {}, "floor skipped: no benchmark measured on both sides")
    if regressed:
        worst = ", ".join(f"{n} -{drops[n]:.3f} (tol {tolerances[n]:.3f})" for n in regressed)
        return FloorVerdict(False, regressed, drops, tolerances,
                            f"floor vetoed on {len(regressed)} benchmark(s): {worst}")
    held = ", ".join(f"{n} {-drops[n]:+.3f}" for n in sorted(drops))
    return FloorVerdict(True, [], drops, tolerances, f"floor held across {len(drops)}: {held}")


_PROMOTE, _ARCHIVE, _REJECT = "promote", "archive", "reject"


def apply_regression_floor(base_verdict: str, fv: FloorVerdict) -> tuple[str, str]:
    """Fold the floor into the in-domain verdict. Can only downgrade, never promote --
    mirrors :func:`cross_bench.apply_cross_bench` so the two gates compose predictably."""
    base = (base_verdict or "").strip().lower()
    if fv.passed:
        return base, f"mixture floor ok; {fv.detail}"
    if base == _PROMOTE:
        return _ARCHIVE, f"mixture floor capped promote->archive; {fv.detail}"
    if base == _ARCHIVE:
        return _ARCHIVE, f"mixture floor regressed but kept as stepping stone; {fv.detail}"
    return _REJECT, f"mixture floor regressed and no in-domain promotion; {fv.detail}"


def mixture_k(default: int = 1) -> int:
    """Samples per task when scoring the mixture (``DARWINX_GATE_MIXTURE_K``).

    Cost here is multiplicative in the number of benchmarks, so the default is one pass. The
    floor's tolerance is denominated in the *baseline's* replicate sd, which already accounts
    for single-pass noise, so k=1 is honest -- just noisy.
    """
    try:
        v = int(os.environ.get("DARWINX_GATE_MIXTURE_K", "").strip() or default)
    except ValueError:
        return default
    return v if v > 0 else default


#: Spec ``eval`` keys and the CodingBenchEvalConfig fields they set. Every one of them is reset
#: for a component that does not name it, rather than left at the primary benchmark's value —
#: inheriting them is the bug this mapping exists to prevent.
_EVAL_FIELDS = {
    "split": "benchmark_split",
    "namespace": "benchmark_namespace",
    "tag": "benchmark_tag",
    "registry": "benchmark_registry",
    "options": "benchmark_options",
}


def _component_cfg(base_cb_cfg, comp: BenchComponent):
    """``base_cb_cfg`` re-pointed at one member of the mixture.

    Only the benchmark identity is taken from the component; everything else -- model, agent,
    turn budget, runtime, parallelism -- stays as the campaign configured it, because those are
    what the candidate is being measured *with* and must not vary between members.

    Unknown fields are skipped rather than passed to ``replace``, so a spec written by a newer
    calibrator against an older driver degrades to "this knob had no effect" instead of a
    TypeError that takes the whole floor out.
    """
    updates: dict = {
        "benchmark_name": comp.name,
        "dataset": comp.dataset or comp.eval.get("dataset")
                   or getattr(base_cb_cfg, "dataset", None),
    }
    for key, attr in _EVAL_FIELDS.items():
        if not hasattr(base_cb_cfg, attr):
            continue
        value = comp.eval.get(key)
        updates[attr] = dict(value) if key == "options" and isinstance(value, dict) else value
    return replace(base_cb_cfg, **updates)


@dataclass(frozen=True)
class MixtureMeasurement:
    """What the gate measured, and on how many tasks.

    The sizes travel with the rates rather than being recomputed, because the floor's tolerance
    depends on them: a screen of 25 tasks and a full sample of 80 give the same kind of number
    with three times the noise, and a tolerance derived from the wrong one either vetoes healthy
    candidates or stops vetoing at all.
    """

    rates: dict[str, float] = field(default_factory=dict)
    n_used: dict[str, int] = field(default_factory=dict)


def run_mixture_stage(
    base_cb_cfg,
    spec: MixtureSpec,
    *,
    stage: str = "full",
    only: list[str] | None = None,
    k_samples: int | None = None,
    config_path=None,
    cwd=None,
    extra_env: dict[str, str] | None = None,
    tee_log_path=None,
) -> MixtureMeasurement:
    """Score the mixture at one stage of the floor.

    ``stage="screen"`` measures the cheap prefix sample, ``"full"`` the whole gate sample.
    ``only`` restricts the pass to named benchmarks, which is how escalation stays cheap: after a
    screen flags one benchmark, only that one is re-measured at full size.
    """
    from dataclasses import replace

    from . import codingbench_eval as cbe

    k = k_samples or mixture_k()
    rates: dict[str, float] = {}
    n_used: dict[str, int] = {}
    wanted = set(only) if only else None

    for name in spec.names:
        if wanted is not None and name not in wanted:
            continue
        comp = spec.components[name]
        if not comp.tasks:
            continue

        gate_tasks = comp.gate_screen_sample() if stage == "screen" else comp.gate_sample()
        if not gate_tasks:
            continue

        per_task: dict[str, float] = {}
        if _bridge_handles(name):
            per_task = _bridge_rates(
                name, comp, base_cb_cfg, gate_tasks, k_samples=k, cwd=cwd,
                tee_log_path=tee_log_path,
            )
        else:
            cfg = _component_cfg(base_cb_cfg, comp)
            res = cbe.run_subset_sampled(
                cb_cfg=cfg,
                task_names=gate_tasks,
                k_samples=k,
                config_path=config_path,
                cwd=cwd,
                extra_env=extra_env,
                tee_log_path=tee_log_path,
            )
            per_task = {t: rate for t, (rate, _n) in res.rates.items()}

        if per_task:
            rates[name] = sum(per_task.values()) / len(per_task)
            # The number of tasks that actually produced a measurement, not the number asked
            # for: an outage that loses half the sample makes the measurement noisier, and the
            # tolerance has to know.
            n_used[name] = len(per_task)

    return MixtureMeasurement(rates=rates, n_used=n_used)


def run_mixture_rates(
    base_cb_cfg,
    spec: MixtureSpec,
    *,
    k_samples: int | None = None,
    config_path=None,
    cwd=None,
    extra_env: dict[str, str] | None = None,
    tee_log_path=None,
) -> dict[str, float]:
    """Score the candidate on every benchmark in the mixture: name -> pass rate.

    One eval pass per benchmark, because the eval seam is single-benchmark all the way down
    (one harness, one dataset, one grader). A benchmark that yields no measurable task is
    omitted rather than recorded as zero: absent evidence is not evidence of failure, and a
    fabricated 0.0 would trip the regression floor on an infrastructure outage.

    Two backends, chosen per benchmark. Most benchmarks run on the legacy runner. Ones that
    require network isolation cannot -- the vendored xrlenv has no pier support, so the trial
    dies at network setup -- and those route through :mod:`beagle_bridge`, which drives them
    through beagle's pier harness. The choice is per benchmark rather than global because the
    legacy runner is what the baselines for SWE-V and Pro were measured on, and moving a
    benchmark to a different backend than its own baseline would make its sigmas meaningless.

    Rates only. Callers that need the tolerance to match the sample size -- which is every
    caller running the two-stage floor -- want :func:`run_mixture_stage` instead.
    """
    return run_mixture_stage(
        base_cb_cfg, spec, stage="full", k_samples=k_samples, config_path=config_path,
        cwd=cwd, extra_env=extra_env, tee_log_path=tee_log_path,
    ).rates


def _bridge_handles(benchmark: str) -> bool:
    """Whether this benchmark must be scored through beagle.

    Import-guarded: the bridge is optional, and a driver running where beagle is not installed
    should fall through to the legacy runner rather than fail to import.
    """
    try:
        from . import beagle_bridge
    except ImportError:
        return False
    return beagle_bridge.handles(benchmark)


def _bridge_rates(
    benchmark: str,
    comp,
    base_cb_cfg,
    task_names: list[str],
    *,
    k_samples: int,
    cwd=None,
    job_name: str | None = None,
    tee_log_path=None,
) -> dict[str, float]:
    """Per-task pass rates from the beagle backend, or ``{}`` if it could not measure.

    Returning empty on failure is deliberate: the caller omits an unmeasurable benchmark from
    the mixture, which is strictly better than a fabricated 0.0 that would trip the regression
    floor on what is really an infrastructure problem.

    WHY THIS PUBLISHES A COMMIT INSTEAD OF READING ``cb_cfg.monet_ref``
      A trial container clones the agent; it cannot see the node's worktree. The legacy runner
      handles this *inside* ``run_subset_sampled``: it publishes the worktree's HEAD as a
      throwaway branch and rewrites the ref. The config it is handed still carries the default
      ref, so reading that field here would have scored ``main`` on every candidate -- and
      ``main`` scores the baseline by construction, so this benchmark would have sat inside the
      floor reporting no change forever, never vetoing anything, while the log showed a
      three-benchmark gate. A benchmark that cannot fail is worse than an absent one, because
      it is counted.

      So the bridge does the same publish/delete cycle explicitly, and refuses to measure when
      it has no candidate ref of its own.
    """
    from . import beagle_bridge
    from . import codingbench_eval as cbe

    cfg, candidate = cbe._maybe_inject_candidate_commit(
        cwd=cwd, job_name=job_name, cb_cfg=base_cb_cfg,
    )
    if candidate is None:
        # No published candidate: either injection is off or there is no worktree here. The
        # only ref available would be the default one, which is the baseline. Omitting the
        # benchmark loses a signal; scoring the baseline as the candidate invents one.
        return {}

    try:
        result = beagle_bridge.run_subset(
            benchmark,
            list(task_names),
            repo_url=cfg.monet_repo_url,
            ref=cfg.monet_ref,
            k_samples=k_samples,
            tee_log_path=tee_log_path,
        )
        return {task: rate for task, (rate, _n) in result.rates.items()}
    finally:
        # Throwaway branches accumulate on the remote otherwise: one per benchmark per
        # promotion attempt, which over a 100-node campaign is hundreds of dead refs.
        cbe._delete_candidate_ref(candidate)


__all__ = [
    "DEFAULT_SD_FLOOR", "DEFAULT_TOL_SD", "DEFAULT_GAIN_CAP_SD",
    "DEFAULT_GATE_TASKS", "DEFAULT_GATE_SCREEN_TASKS",
    "gate_enabled", "tolerance_sd", "gain_cap_sd", "min_abs_drop", "mixture_k",
    "gate_tasks_n", "gate_screen_n", "gate_seed",
    "MixtureMeasurement", "run_mixture_stage",
    "BenchComponent", "MixtureSpec", "load_mixture_spec",
    "MixtureFitness", "normalized_gain", "mixture_fitness",
    "FloorVerdict", "regression_floor_verdict", "apply_regression_floor",
    "run_mixture_rates",
]
