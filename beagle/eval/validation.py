"""Validation signals — turning any benchmark's results into one comparable feedback channel.

An evolution loop is only as good as the signal it selects on, and a signal that means
different things on different benchmarks is worse than useless: it looks like a number, so it
gets averaged. This module is the alignment layer. Whatever the benchmark measures -- a patch
that applies and passes tests, a terminal reward, tokens burned, turns taken -- it lands here
as a :class:`Measurement`, gets aligned against a :class:`Reference` into signed sigma where
**positive always means better**, and comes out as a :class:`ValReport` the loop can select on
and the proposer can read.

Three properties are the point, and each one exists because its absence has already cost us:

**Not measuring is not failing.** A task whose container never came up, whose gateway 401'd,
or whose runner was killed produced *no measurement*. Scoring it as a loss silently converts
infrastructure trouble into evolutionary pressure -- the loop learns from our outages. Every
count here splits three ways (solved / failed / unmeasured), aggregates ignore the third, and
coverage is reported so a run that measured a quarter of its tasks cannot pass itself off as a
result.

**Direction is part of the metric.** Pass rate goes up, tokens and turns go down. Once
alignment flips lower-is-better into the same signed-sigma space, "the harness got 30% more
compact" and "DeepSWE went up two points" are finally the same kind of number, and the
consolidation pressure stops needing its own parallel scoring path.

**Comparability requires calibration, and calibration can be missing.** A raw delta means
nothing without knowing how much that benchmark moves between identical runs. A slice with no
:class:`Reference` is marked uncalibrated and kept out of aggregates rather than being handed
a default sigma -- an invented yardstick is the one failure mode that would be invisible.

The input is ``list[TaskResult]``, which every benchmark's runner already produces and which
already carries ``benchmark``, so no per-benchmark adapter is needed for the common case; a
benchmark with an unusual error taxonomy supplies an :class:`OutcomePolicy` instead of a new
code path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from beagle.types import TaskResult

#: Divisor floor for sigma. A slice whose measured noise is ~0 would otherwise turn a rounding
#: difference into an unbounded signal. 0.01 = one point of a rate.
DEFAULT_SD_FLOOR = 0.01


class Direction(str, Enum):
    """Which way is better. Alignment uses this to put every metric in one signed space."""

    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"


class Outcome(str, Enum):
    """What a single trial actually told us.

    ``UNMEASURED`` is the load-bearing one: it is not a third grade of badness, it is the
    absence of evidence, and it must never reach an aggregate.
    """

    SOLVED = "solved"
    FAILED = "failed"
    UNMEASURED = "unmeasured"


# ── outcome classification ───────────────────────────────────────────────

#: Error substrings that mean "we failed to measure", not "the agent failed". Deliberately
#: about the harness and the plumbing, never about the agent's own behaviour.
DEFAULT_UNMEASURED_MARKERS: tuple[str, ...] = (
    "ConnectionError", "ConnectError", "ProtocolError", "RemoteDisconnected",
    "ImageNotFound", "ContainerError", "DockerException", "PermissionError",
    "HTTP 401", "HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504",
    "gateway", "Gateway", "tunnel", "APIConnectionError", "RateLimit",
    "InstallError", "SetupError", "harness", "Harness",
)


@dataclass(frozen=True)
class OutcomePolicy:
    """How to read a :class:`TaskResult`. Defaults encode our strict scoring rule.

    ``timeout_is_failure`` is a real judgement call, not a detail. On TB2.1, 17 trials passed
    the verifier at reward 1.0 and then raised ``AgentTimeoutError``; counting those as losses
    is defensible (the agent did not finish in budget) but it distributed unevenly across
    agents -- 7/5/3/2 -- and so moved exactly the between-agent comparison it was supposed to
    inform. Whichever way it is set, it must be set explicitly and identically for baseline
    and candidate, which is why it lives in a policy object rather than in an ``if``.
    """

    unmeasured_markers: tuple[str, ...] = DEFAULT_UNMEASURED_MARKERS
    #: Timeouts are the agent's problem (a budget it blew), not the harness's.
    timeout_is_failure: bool = True
    timeout_markers: tuple[str, ...] = ("Timeout", "timeout", "TimeoutError")
    #: For in-band-graded benchmarks: reward at or above this counts as solved.
    reward_threshold: float = 1.0
    #: Strict rule: an errored trial is not a win even if the grader liked it.
    error_voids_success: bool = True


def classify(result: TaskResult, policy: OutcomePolicy | None = None) -> Outcome:
    """Decide whether a trial was solved, failed, or never actually measured."""
    policy = policy or OutcomePolicy()
    err = (result.error or "").strip()

    if err:
        is_timeout = any(m in err for m in policy.timeout_markers)
        if is_timeout:
            # A timeout is a measurement: the agent ran and did not finish in budget.
            if policy.timeout_is_failure:
                return Outcome.FAILED
        elif any(m in err for m in policy.unmeasured_markers):
            return Outcome.UNMEASURED
        elif policy.error_voids_success:
            # An unrecognised error is treated as unmeasured rather than as a loss. Guessing
            # wrong in this direction costs a data point; guessing wrong the other way feeds
            # our own breakage back into selection.
            return Outcome.UNMEASURED

    if result.resolved:
        return Outcome.SOLVED
    if result.reward is not None and float(result.reward) >= policy.reward_threshold:
        return Outcome.SOLVED if not (err and policy.error_voids_success) else Outcome.FAILED
    return Outcome.FAILED


# ── measurements ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Measurement:
    """One scalar observation of one aspect of a candidate, with its provenance."""

    name: str
    value: float
    direction: Direction
    #: Distinct tasks that produced a measurement.
    n: int
    #: Trials per task actually observed (avg@k's k), reported as the max seen.
    k: int = 1
    #: Trials discarded as UNMEASURED. Not failures.
    unmeasured: int = 0
    unit: str = ""

    @property
    def coverage(self) -> float:
        """Share of attempted trials that produced a measurement."""
        total = self.n * max(self.k, 1) + self.unmeasured
        return (self.n * max(self.k, 1) / total) if total else 0.0


#: Builds a Measurement from the measured (non-UNMEASURED) results of one slice.
MetricExtractor = Callable[[Sequence[tuple[TaskResult, Outcome]]], "Measurement | None"]


def _by_task(graded: Sequence[tuple[TaskResult, Outcome]]) -> dict[str, list[tuple[TaskResult, Outcome]]]:
    out: dict[str, list[tuple[TaskResult, Outcome]]] = {}
    for r, o in graded:
        out.setdefault(str(r.task_id), []).append((r, o))
    return out


def pass_rate(graded: Sequence[tuple[TaskResult, Outcome]]) -> Measurement | None:
    """avg@k: per-task solve rate over measured trials, then the mean across tasks.

    Averaging per task first rather than over all trials keeps a task that happened to run
    more times from carrying more weight -- with retries in the mix, k is rarely uniform.
    """
    unmeasured = sum(1 for _, o in graded if o is Outcome.UNMEASURED)
    per_task = _by_task([(r, o) for r, o in graded if o is not Outcome.UNMEASURED])
    if not per_task:
        return None
    rates = [
        sum(1 for _, o in trials if o is Outcome.SOLVED) / len(trials)
        for trials in per_task.values()
    ]
    return Measurement(
        name="pass_rate",
        value=sum(rates) / len(rates),
        direction=Direction.HIGHER_IS_BETTER,
        n=len(per_task),
        k=max(len(t) for t in per_task.values()),
        unmeasured=unmeasured,
        unit="rate",
    )


def _mean_of(graded, name, pick, unit) -> Measurement | None:
    """Mean of a lower-is-better per-trial quantity over measured trials."""
    unmeasured = sum(1 for _, o in graded if o is Outcome.UNMEASURED)
    vals: list[float] = []
    tasks: set[str] = set()
    for r, o in graded:
        if o is Outcome.UNMEASURED:
            continue
        v = pick(r)
        if v is None:
            continue
        vals.append(float(v))
        tasks.add(str(r.task_id))
    if not vals:
        return None
    return Measurement(name=name, value=sum(vals) / len(vals),
                       direction=Direction.LOWER_IS_BETTER, n=len(tasks),
                       k=max(1, round(len(vals) / max(len(tasks), 1))),
                       unmeasured=unmeasured, unit=unit)


def mean_turns(graded) -> Measurement | None:
    return _mean_of(graded, "mean_turns", lambda r: r.num_turns or None, "turns")


def mean_duration_sec(graded) -> Measurement | None:
    return _mean_of(graded, "mean_duration_sec", lambda r: r.duration_sec or None, "seconds")


def mean_tokens(graded) -> Measurement | None:
    def total(r: TaskResult):
        vals = [v for v in (r.tokens or {}).values() if isinstance(v, (int, float))]
        return sum(vals) if vals else None
    return _mean_of(graded, "mean_tokens", total, "tokens")


#: Correctness plus the efficiency metrics the consolidation pressure needs. Both kinds go
#: through the same alignment, which is the whole point.
DEFAULT_METRICS: tuple[MetricExtractor, ...] = (
    pass_rate, mean_turns, mean_duration_sec, mean_tokens,
)


# ── alignment ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Reference:
    """What this metric was worth before, and how much it moves on its own.

    ``sd`` must come from replicate runs of the *same* harness, not from variation across
    candidates -- otherwise the yardstick stretches with the thing being measured.
    """

    baseline: float
    sd: float = 0.0
    n: int = 0


@dataclass(frozen=True)
class AlignedSignal:
    """A measurement expressed as signed sigma, positive = better, whatever the metric."""

    slice_id: str
    metric: str
    sigma: float | None          # None when uncalibrated
    value: float
    baseline: float | None
    direction: Direction
    n: int
    k: int
    unmeasured: int
    unit: str

    @property
    def calibrated(self) -> bool:
        return self.sigma is not None


def align(slice_id: str, m: Measurement, ref: Reference | None, *,
          sd_floor: float = DEFAULT_SD_FLOOR) -> AlignedSignal:
    """Put one measurement into the common signed-sigma space.

    Lower-is-better metrics have their delta negated, so a harness that got cheaper and one
    that got more correct both read positive and can be weighed against each other.
    """
    sigma: float | None = None
    if ref is not None:
        delta = m.value - ref.baseline
        if m.direction is Direction.LOWER_IS_BETTER:
            delta = -delta
        sigma = delta / max(ref.sd, sd_floor)
    return AlignedSignal(
        slice_id=slice_id, metric=m.name, sigma=sigma, value=m.value,
        baseline=(ref.baseline if ref else None), direction=m.direction,
        n=m.n, k=m.k, unmeasured=m.unmeasured, unit=m.unit,
    )


# ── the report ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValReport:
    """Aligned signals for every slice, plus what could not be measured or calibrated."""

    signals: list[AlignedSignal] = field(default_factory=list)
    #: Slices that produced no measurement at all (every trial unmeasured).
    empty_slices: list[str] = field(default_factory=list)

    @property
    def slices(self) -> list[str]:
        return sorted({s.slice_id for s in self.signals})

    def get(self, slice_id: str, metric: str = "pass_rate") -> AlignedSignal | None:
        for s in self.signals:
            if s.slice_id == slice_id and s.metric == metric:
                return s
        return None

    def values(self, metric: str = "pass_rate") -> dict[str, float]:
        """``{slice: raw value}`` -- what the aggregate and the floor consume."""
        return {s.slice_id: s.value for s in self.signals if s.metric == metric}

    def sigmas(self, metric: str = "pass_rate") -> dict[str, float]:
        """``{slice: signed sigma}``, calibrated slices only."""
        return {s.slice_id: s.sigma for s in self.signals
                if s.metric == metric and s.sigma is not None}

    def uncalibrated(self, metric: str = "pass_rate") -> list[str]:
        return sorted(s.slice_id for s in self.signals
                      if s.metric == metric and s.sigma is None)

    def low_coverage(self, metric: str = "pass_rate", *, threshold: float = 0.8) -> list[str]:
        """Slices where too much went unmeasured to trust the number."""
        out = []
        for s in self.signals:
            if s.metric != metric:
                continue
            attempted = s.n * max(s.k, 1) + s.unmeasured
            if attempted and (s.n * max(s.k, 1)) / attempted < threshold:
                out.append(s.slice_id)
        return sorted(out)

    def to_feedback(self, metric: str = "pass_rate") -> str:
        """A compact human/proposer-readable summary.

        The evolver gets told *where* it lost, not just that it did -- a gate returns a bit,
        and a bit is not something a proposer can act on.
        """
        lines: list[str] = []
        for slice_id in self.slices:
            s = self.get(slice_id, metric)
            if s is None:
                continue
            if s.sigma is None:
                lines.append(f"  {slice_id}: {s.value:.3f} {s.unit} (uncalibrated, not scored)")
            else:
                arrow = "better" if s.sigma > 0 else "worse" if s.sigma < 0 else "flat"
                lines.append(
                    f"  {slice_id}: {s.value:.3f} {s.unit} "
                    f"({s.sigma:+.2f}sigma {arrow} vs {s.baseline:.3f}, n={s.n} k={s.k}"
                    + (f", {s.unmeasured} unmeasured" if s.unmeasured else "") + ")"
                )
        if self.empty_slices:
            lines.append(f"  no measurement at all: {', '.join(self.empty_slices)}")
        thin = self.low_coverage(metric)
        if thin:
            lines.append(f"  coverage too thin to trust: {', '.join(thin)}")
        return "\n".join(lines) if lines else "  (no validation signals)"


def summarize(
    results: Iterable[TaskResult],
    *,
    references: dict[str, Reference] | dict[tuple[str, str], Reference] | None = None,
    policy: OutcomePolicy | None = None,
    metrics: Sequence[MetricExtractor] = DEFAULT_METRICS,
    slice_of: Callable[[TaskResult], str] | None = None,
    sd_floor: float = DEFAULT_SD_FLOOR,
) -> ValReport:
    """Group results into slices, measure each, and align them into one report.

    Slices default to ``TaskResult.benchmark``, which every runner already sets, so a mixture
    needs no extra bookkeeping to be scored per benchmark. ``slice_of`` overrides that for
    finer cuts (task family, difficulty band) without touching anything upstream.

    ``references`` may be keyed by slice (applying to that slice's primary metric) or by
    ``(slice, metric)`` for full control.
    """
    policy = policy or OutcomePolicy()
    slice_of = slice_of or (lambda r: r.benchmark or "unknown")
    refs = references or {}

    grouped: dict[str, list[tuple[TaskResult, Outcome]]] = {}
    for r in results:
        grouped.setdefault(slice_of(r), []).append((r, classify(r, policy)))

    signals: list[AlignedSignal] = []
    empty: list[str] = []
    for slice_id in sorted(grouped):
        graded = grouped[slice_id]
        produced = False
        for extractor in metrics:
            m = extractor(graded)
            if m is None:
                continue
            produced = True
            ref = refs.get((slice_id, m.name)) or (  # type: ignore[call-overload]
                refs.get(slice_id) if m.name == "pass_rate" else None  # type: ignore[call-overload]
            )
            signals.append(align(slice_id, m, ref, sd_floor=sd_floor))
        if not produced:
            empty.append(slice_id)
    return ValReport(signals=signals, empty_slices=empty)


def capture(evaluate: Callable[[object], None], candidate) -> list[TaskResult]:
    """Run an ``Evaluate`` and return its results without disturbing the candidate.

    ``Trainer.fit`` builds ``val`` with the same ``_make_evaluate`` as the training scorer, so
    both write ``candidate.score`` and ``candidate.results`` in place. Calling ``val`` after
    ``evaluate`` therefore overwrites the fitness that was just measured with the held-out
    numbers -- the candidate ends up carrying val results labelled as its score, and the two
    are indistinguishable afterwards. Snapshot and restore so a caller can hold both.
    """
    prior_score = getattr(candidate, "score", None)
    prior_results = list(getattr(candidate, "results", []) or [])
    try:
        evaluate(candidate)
        return list(getattr(candidate, "results", []) or [])
    finally:
        candidate.score = prior_score
        candidate.results = prior_results


__all__ = [
    "DEFAULT_SD_FLOOR", "DEFAULT_UNMEASURED_MARKERS", "DEFAULT_METRICS",
    "capture",
    "Direction", "Outcome", "OutcomePolicy", "classify",
    "Measurement", "MetricExtractor",
    "pass_rate", "mean_turns", "mean_duration_sec", "mean_tokens",
    "Reference", "AlignedSignal", "align",
    "ValReport", "summarize",
]
