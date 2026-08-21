"""MatchFixGate — equivalence-based monotonic-improvement gate.

The post-hoc statistical gates (L4 path-allowlist, L5 jaccard-credibility,
parent-credibility weighting) recorded regression catastrophes but could not
prevent them — they ran *after* the full final-eval. This module replaces
those gates with a behavioral check that runs *before* final-eval:

1. ``analyze_diff``     — LLM reads the diff + trial digests, names the
                          behavioral surfaces the change touches.
2. ``select_probe_tasks`` — LLM picks K tasks from ``parent.solved_tasks``
                          most likely affected by those surfaces.
3. ``execute_probes``   — Run those K tasks against the child harness via
                          a Protocol-shaped Harbor runner (~15-25 min).
4. ``verdict``          — LLM consumes the probe results and labels the
                          change ``EQUIVALENT`` / ``MODIFIED`` /
                          ``INCONCLUSIVE``.
5. ``extension_check``  — Pure set arithmetic. Did the child solve at
                          least one task the parent left unsolved?

Inspired by MatchFixAgent (arXiv:2509.16187) — the paper decomposes
equivalence validation into focused single-purpose sub-agents and reports
99.2 % verdict accuracy versus a monolithic prompt's 18.5 %.

Each function is a pure function over Protocols (``LLMBackend``,
``HarborRunner``); no pipeline-state coupling. The pipeline integration
lives in ``self_evolve.atelier_hook.run_equivalence_gate``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


_LOG = logging.getLogger("atelier.matchfix_gate")


# ─── Regression-tolerance relaxations (anti-false-negative) ─────────────────
# The legacy gate is "super strict": a SINGLE probed regression (incl. a 1-in-k
# sampling blip), a single dissenting LLM panelist, or an LLM/infra hiccup all
# force a confirmed MODIFIED — the only hard-reject — which skips the full eval
# entirely. Campaign mining showed this discards most improved-but-regressed
# candidates (the highest-potential nodes). These knobs let the gate behave as a
# CHEAP filter for clearly net-negative changes while deferring borderline ones
# to the avg@k full eval (the ground-truth arbiter) instead of hard-killing them.
# ALL default to the legacy strict behaviour, so an unset environment reproduces
# the control arm byte-for-byte.


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _gate_regression_tol() -> int:
    """Max # of probe regressions to TOLERATE at the gate (downgrade a
    MODIFIED to INCONCLUSIVE so the full eval arbitrates). 0 = legacy strict."""
    try:
        return max(0, int(os.environ.get("ATELIER_GATE_REGRESSION_TOL", "0")))
    except ValueError:
        return 0


def _apply_gate_tolerance(
    decision: str,
    probe_results: "ProbeResults",
    extension: "ExtensionResult | None",
) -> tuple[str, str]:
    """Downgrade a tolerable MODIFIED verdict to INCONCLUSIVE.

    A MODIFIED is the pipeline's only hard-reject and it SKIPS the full eval, so
    a candidate that improves many tasks but trips a couple of probes is killed
    before the ground-truth test runs. When ``ATELIER_GATE_REGRESSION_TOL`` > 0
    we tolerate up to ``tol`` probe regressions (plus one extra per newly-solved
    EXTENSION task — net-value), turning the verdict INCONCLUSIVE so it falls
    through to the avg@k full eval. The full eval + post-final extension gate
    remain the real arbiters, and the avg@k parent-selection still prevents a
    net-worse agent from being promoted, so this cannot cause silent drift.
    Returns ``(decision, note)``; ``note`` is "" when nothing changed.
    """
    if decision != "MODIFIED":
        return decision, ""
    tol = _gate_regression_tol()
    if tol <= 0:
        return decision, ""
    n_fail = probe_results.n_fail
    n_pass = probe_results.n_pass
    n_ext = len(extension.extension_tasks_solved) if extension else 0
    allow = n_fail <= tol or (n_ext > 0 and n_fail <= tol + n_ext)
    if not allow:
        return decision, ""
    note = (
        f"gate-tolerance: {n_fail} probe regression(s) within tol={tol} "
        f"(n_pass={n_pass}, extensions={n_ext}); downgrading MODIFIED→"
        f"INCONCLUSIVE to defer to the full eval (ground-truth arbiter)"
    )
    _LOG.info(note)
    return "INCONCLUSIVE", note


# ─── Protocols ──────────────────────────────────────────────────────────────


class LLMBackend(Protocol):
    """Minimal chat-completion protocol the gate's three LLM-using stages
    depend on.

    Implementations:
      - ``OpenAIChatBackend`` (production) — see ``atelier.matchfix_gate``
        bottom of file.
      - ``FakeLLMBackend`` (tests) — fixture in
        ``tests/atelier/test_matchfix_gate.py``.

    The single ``complete()`` method takes a prompt string + optional
    ``system`` and returns plain text. Free-text return; callers parse
    JSON / labels themselves (keeps the protocol surface small and
    testable).
    """

    def complete(self, *, prompt: str, system: str | None = None) -> str: ...


class HarborRunner(Protocol):
    """Minimal protocol for invoking Harbor on K probe tasks against a
    specific child commit.

    Implementations:
      - production: a thin wrapper around
        ``self_evolve.eval_runner.run_subset`` that arranges the worktree
        is checked out at ``child_commit_sha`` first.
      - tests: a fake that returns canned per-task reward dicts.

    The ``run`` method may return either:
    - ``Mapping[str, float]`` — bare rewards, no failure digests, OR
    - ``(Mapping[str, float], Mapping[str, str])`` — rewards + first-
      failure verifier-stdout digests keyed by task id (per §4.2.1 of
      the harness survey; MAGE-style waveform-window evidence).

    ``execute_probes`` accepts either shape so legacy runners and
    fakes don't need to change.
    """

    def run(
        self,
        *,
        child_commit_sha: str,
        probe_tasks: Sequence[str],
    ) -> Any: ...


# ─── Dataclasses ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SemanticAnalysis:
    """Output of stage 1 — what the diff touches, in behavior terms."""

    modified_surfaces: tuple[str, ...]
    """Short identifiers of behavioral surfaces the diff modifies. E.g.
    ``("evidence_classifier", "system_prompt")``."""

    risk_level: str
    """``low`` (e.g. added bundled skills) / ``medium`` (modified tool
    dispatcher) / ``high`` (touched query/loop). Used as a tie-break in
    the verdict — high-risk surfaces force more conservative verdicts."""

    rationale: str
    """One paragraph from the LLM — why these surfaces, why this
    risk_level. Surfaced in the sidecar for human auditing."""

    raw_response: str = ""
    """Raw LLM reply (useful for debugging parser drift)."""


@dataclass(frozen=True)
class ProbeResults:
    """Output of stage 3 — Harbor execution of the K probe tasks."""

    probe_tasks: tuple[str, ...]
    rewards: Mapping[str, float]
    """``{task_id: reward}`` ∈ [0, 1]. Missing tasks default to 0.0 when
    used in derivations below."""

    failure_digests: Mapping[str, str] = field(default_factory=dict)
    """Optional first-failure digests for tasks that regressed. Each
    entry is the first ~500 chars of the failing trial's verifier
    stdout — the harness equivalent of MAGE's "waveform window around
    the first failing clock cycle" (arXiv 2605.18747 §4.2.1).

    Empty for trivial-pass paths or when the runner doesn't supply
    them. Surfaced in the verdict prompt so the LLM can attribute
    failures concretely instead of speculating."""

    adversarial_picks: tuple[str, ...] = ()
    """Subset of probe_tasks that came from the adversarial selector
    (i.e., NOT from the LLM's "most likely affected" pick). When one of
    these regresses, the diff's blast radius was larger than the
    semantic_analyzer claimed — a stronger signal than an "affected"
    probe regressing."""

    @property
    def n_pass(self) -> int:
        return sum(1 for t in self.probe_tasks if self.rewards.get(t, 0.0) >= 1.0)

    @property
    def n_fail(self) -> int:
        return len(self.probe_tasks) - self.n_pass

    @property
    def regressed_tasks(self) -> tuple[str, ...]:
        """Probe tasks that did NOT fully pass. Since probes are drawn
        from ``parent.solved_tasks`` (i.e., parent already passed each
        one), any non-pass here is a regression."""
        return tuple(t for t in self.probe_tasks if self.rewards.get(t, 0.0) < 1.0)

    @property
    def n_adversarial_regressed(self) -> int:
        """How many regressions came from adversarial (vs LLM-picked)
        probes. Reading: high count ⇒ semantic_analyzer under-claimed
        the modified surfaces."""
        regressed = set(self.regressed_tasks)
        return sum(1 for t in self.adversarial_picks if t in regressed)


@dataclass(frozen=True)
class ExtensionResult:
    """Output of stage 5 — pure set arithmetic, no LLM."""

    extension_tasks_solved: tuple[str, ...]
    """Tasks the child solved that the parent did NOT.
    ``|child_solved ∩ parent_unsolved|``."""

    has_extension: bool
    """True iff ``len(extension_tasks_solved) >= 1``."""


@dataclass(frozen=True)
class VerificationScope:
    """What was actually verified, what was NOT, and with what
    confidence (arXiv 2605.18747 §5.2.2 — "evidence bundle").

    The plain `EquivalenceVerdict.decision` says yes/no. This
    dataclass says *what backed the yes/no* — which surfaces the
    probes exercised, which surfaces remain unverified, and what
    confidence the K/|parent.solved| ratio yields.

    Surfaced in the sidecar so downstream consumers (LTM, parent
    selection, retrospective analysis) can read the gate's reasoning
    instead of treating it as opaque."""

    verified_surfaces: tuple[str, ...]
    """Behavioral surfaces from ``semantic_analysis.modified_surfaces``
    that at least one probe task is known to exercise. Today we
    populate this conservatively (all of modified_surfaces, if probes
    came from parent.solved); a future enhancement maps tasks → surfaces
    explicitly."""

    unverified_surfaces: tuple[str, ...]
    """Surfaces named by the analyzer but NOT exercised by any probe
    in this gate run. Empty when all modified surfaces had at least
    one probe touching them."""

    probe_coverage: float
    """K / |parent.solved| — fraction of the parent's solved-tasks
    pool the gate actually verified. 1.0 means we probed every solved
    task (only happens when |parent.solved| ≤ K)."""

    untested_assumption: str | None = None
    """Free-text describing the assumption the gate is implicitly
    making (e.g., "tool dispatcher behavior on the K-out-of-N
    unprobed tasks is the same as on the K probed tasks"). Recorded
    for auditability."""


@dataclass(frozen=True)
class ChangeContract:
    """The 'change contract' the gate emits per accepted/rejected
    candidate (arXiv 2605.18747 §5.2.3).

    The paper's prescription, verbatim:

      'Every proposed edit should carry a change contract: which
       component is modified, which failure mode it targets, what
       improvement it predicts, which invariants it must preserve,
       which evaluation can falsify it, and how it can be rolled back.'

    This dataclass operationalizes that — every gate sidecar carries
    one, so a retrospective audit can ask "was the contract met?"
    rather than guessing.
    """

    invariants_preserved: tuple[str, ...]
    """Concrete invariants the gate verified (or attempted to
    verify). The canonical one for MatchFixGate:
    'parent.solved ⊆ child.solved on K probe tasks'."""

    falsifier: tuple[str, ...]
    """The specific probe task ids that would have to fail to flip
    the verdict from EQUIVALENT to MODIFIED. Same as
    ``probe_results.regressed_tasks`` for MODIFIED verdicts; the set
    of probe tasks whose regression WOULD be a counter-example for
    EQUIVALENT verdicts."""

    rollback_action: str
    """The git command that undoes the candidate. For self_evolve
    pipelines, this is always
    ``git checkout <parent.commit_sha>`` since each child is its own
    branch. Recorded so the contract is auditable end-to-end."""

    modified_component: str
    """Short label naming the component the diff touches. Derived
    from ``semantic_analysis.modified_surfaces`` joined with '+'."""


@dataclass(frozen=True)
class EquivalenceVerdict:
    """The aggregate gate decision for one candidate.

    ``accept_for_final_eval`` is the only field pipeline.py reads — the
    other fields are for sidecar diagnostics + future retrospective
    analysis.
    """

    decision: str
    """One of ``"EQUIVALENT"`` / ``"MODIFIED"`` / ``"INCONCLUSIVE"``.

    - ``EQUIVALENT``: parent-solved tasks remain solved → safe to
      proceed (subject to ``extension_check`` requiring at least one
      new solve).
    - ``MODIFIED``: at least one probe regressed AND verdict_agent
      cannot attribute the regression to noise → reject.
    - ``INCONCLUSIVE``: probes inconclusive (e.g., all errored, or
      verdict_agent couldn't decide). Conservative default: reject."""

    k_picked_tasks: tuple[str, ...]
    per_task_results: Mapping[str, float]
    extension_tasks_solved: tuple[str, ...]
    n_regressions: int

    semantic_analysis: SemanticAnalysis | None
    verdict_rationale: str
    """The LLM's one-paragraph explanation. May be empty when the gate
    short-circuited (e.g., no parent_solved_tasks to probe)."""

    accept_for_final_eval: bool
    """The single bit pipeline.py uses to decide ``run_full`` vs skip."""

    extension_required: bool = True
    """If True, also require ``has_extension`` for acceptance.
    Default True (the monotonic-improvement contract). Disable for
    pure regression-prevention mode without the extension requirement
    (e.g. early calibration runs)."""

    verification_scope: VerificationScope | None = None
    """Evidence bundle: what was verified, what wasn't, confidence
    (§5.2.2). None when scope cannot be computed (e.g., no semantic
    analysis available)."""

    change_contract: ChangeContract | None = None
    """The contract the gate emits per candidate (§5.2.3). None
    when the gate degraded to accept-by-default before any probes."""

    consensus_votes: tuple[str, ...] = ()
    """The individual decisions cast in the multi-vote verdict
    (CANDOR-style aggregation, §4.1.1). For example,
    ``("MODIFIED", "MODIFIED", "INCONCLUSIVE")`` was aggregated to
    ``decision="MODIFIED"`` by majority vote. Empty for the
    short-circuit paths (all probes passed / no probes run / single-
    vote mode)."""

    failure_digests: Mapping[str, str] = field(default_factory=dict)
    """First-failure digests for regressed probe tasks (§4.2.1 MAGE
    pattern). Each entry is the first ~500 chars of the failing
    trial's verifier stdout, enabling concrete attribution in the
    retrospective view rather than just 't2 regressed'."""


# ─── Stage 1: analyze_diff ─────────────────────────────────────────────────


_ANALYZE_SCHEMA_HINT = """Reply with EXACTLY one JSON object (no prose outside it):

```
{
  "modified_surfaces": ["<short_id_1>", "<short_id_2>", ...],
  "risk_level": "low" | "medium" | "high",
  "rationale": "<one paragraph>"
}
```"""


def analyze_diff(
    *,
    diff_text: str,
    trial_digests: str,
    llm: LLMBackend,
    prompt_template: str,
) -> SemanticAnalysis:
    """Stage 1 — name the behavioral surfaces this diff modifies.

    Args:
      diff_text:      ``git diff parent..child`` output (truncate to a
                      few thousand lines before calling).
      trial_digests:  Pre-rendered markdown summarizing the failing
                      trials the proposer was reasoning about (from
                      ``trace_analyzer.render_digest`` / friends).
      llm:            The LLMBackend implementation.
      prompt_template: The matchfix_analyze.md template with two
                      ``{diff}`` / ``{trial_digests}`` placeholders.

    Returns a ``SemanticAnalysis`` even when parsing fails — we degrade
    to ``modified_surfaces=("unknown",)`` + ``risk_level="high"`` so the
    downstream stages default to a conservative posture rather than
    silently dropping into "low risk".
    """
    prompt = (
        prompt_template
        .replace("{diff}", diff_text or "_no diff_")
        .replace("{trial_digests}", trial_digests or "_no digests_")
    )
    prompt = prompt + "\n\n" + _ANALYZE_SCHEMA_HINT

    try:
        raw = llm.complete(prompt=prompt)
    except Exception as e:  # noqa: BLE001 — never crash the gate
        _LOG.warning("analyze_diff LLM call failed: %s", e)
        return SemanticAnalysis(
            modified_surfaces=("unknown",),
            risk_level="high",
            rationale=f"LLM call failed: {e}",
            raw_response="",
        )

    parsed = _extract_json_object(raw)
    if parsed is None:
        _LOG.warning(
            "analyze_diff: could not parse JSON from LLM reply (raw=%r)", raw[:200]
        )
        return SemanticAnalysis(
            modified_surfaces=("unknown",),
            risk_level="high",
            rationale="JSON parse failure; raw reply preserved",
            raw_response=raw,
        )

    surfaces_in = parsed.get("modified_surfaces", []) or []
    surfaces = tuple(str(s).strip() for s in surfaces_in if str(s).strip())
    if not surfaces:
        surfaces = ("unknown",)

    risk_raw = str(parsed.get("risk_level", "high")).strip().lower()
    risk = risk_raw if risk_raw in {"low", "medium", "high"} else "high"

    rationale = str(parsed.get("rationale", "")).strip()

    return SemanticAnalysis(
        modified_surfaces=surfaces,
        risk_level=risk,
        rationale=rationale,
        raw_response=raw,
    )


# ─── Stage 2: select_probe_tasks ────────────────────────────────────────────


_SELECT_SCHEMA_HINT = """Reply with EXACTLY one JSON object (no prose outside it):

```
{
  "probe_tasks": ["<task_id_1>", "<task_id_2>", ...],
  "rationale": "<one short paragraph>"
}
```

The probe_tasks list must:
- be a subset of the candidate_tasks list provided in the prompt,
- contain AT MOST {k} entries,
- be ordered from most-likely-affected first."""


def select_probe_tasks(
    *,
    semantic_analysis: SemanticAnalysis,
    parent_solved_tasks: Sequence[str],
    k: int,
    llm: LLMBackend,
    prompt_template: str,
    n_adversarial: int = 0,
    adversarial_seed: int = 0,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Stage 2 — LLM picks ``k - n_adversarial`` tasks from
    ``parent_solved_tasks`` most likely affected by
    ``semantic_analysis.modified_surfaces``, plus ``n_adversarial``
    tasks chosen deterministically from elsewhere in the pool to
    probe surfaces the LLM did NOT name (arXiv 2605.18747 §4.1.2 —
    adversarial validation).

    The split is the key design choice:
    - LLM picks (= ``k - n_adversarial``) test what the analyzer
      claimed the diff touches → quick way to catch claimed-scope
      regressions.
    - Adversarial picks (= ``n_adversarial``) test surfaces the
      analyzer did NOT name → catch under-claimed blast-radius
      regressions ("the diff said it only touches X but actually
      breaks Y").

    Returns ``(probe_tasks, adversarial_subset)``:
    - probe_tasks: ordered concatenation [LLM-picked..., adversarial...]
    - adversarial_subset: the adversarial picks specifically, so the
      verdict stage can flag them differently if they regress.

    Backward compat: when ``n_adversarial=0``, behavior is identical
    to the original (LLM picks K, no adversarial slot) and the
    returned ``adversarial_subset`` is empty.
    """
    pool = [t for t in dict.fromkeys(parent_solved_tasks) if t]
    if not pool:
        return ((), ())
    if len(pool) <= k:
        return (tuple(pool), ())

    n_adversarial = max(0, min(n_adversarial, k - 1))
    n_llm = k - n_adversarial

    # ─── Stage 2a: LLM picks the "affected" probes ────────────────
    llm_picked: list[str] = []
    if n_llm > 0:
        prompt = (
            prompt_template
            .replace("{modified_surfaces}",
                     ", ".join(semantic_analysis.modified_surfaces))
            .replace("{risk_level}", semantic_analysis.risk_level)
            .replace("{analysis_rationale}",
                     semantic_analysis.rationale or "_none_")
            .replace("{candidate_tasks}", json.dumps(pool))
            .replace("{k}", str(n_llm))
        )
        prompt = prompt + "\n\n" + _SELECT_SCHEMA_HINT.replace("{k}", str(n_llm))

        try:
            raw = llm.complete(prompt=prompt)
        except Exception as e:  # noqa: BLE001
            _LOG.warning("select_probe_tasks LLM call failed: %s", e)
            raw = ""

        parsed = _extract_json_object(raw) if raw else None
        if parsed is not None:
            pool_set = set(pool)
            seen: set[str] = set()
            for t in (parsed.get("probe_tasks", []) or []):
                ts = str(t).strip()
                if not ts or ts not in pool_set or ts in seen:
                    continue
                llm_picked.append(ts)
                seen.add(ts)
                if len(llm_picked) >= n_llm:
                    break

        # Fallback: first-of-pool entries to fill the LLM slot.
        if len(llm_picked) < n_llm:
            for t in pool:
                if t in llm_picked:
                    continue
                llm_picked.append(t)
                if len(llm_picked) >= n_llm:
                    break

    # ─── Stage 2b: adversarial picks from the rest of the pool ────
    # Deterministic shuffle of the *unselected* pool, then pick
    # the first n_adversarial. Determinism (via adversarial_seed) is
    # important so the retrospective can reproduce the gate's
    # behavior; randomness across candidates (via the seed)
    # diversifies coverage over time.
    adversarial_picks: list[str] = []
    if n_adversarial > 0:
        remaining = [t for t in pool if t not in llm_picked]
        # Tiny deterministic shuffle: pick by hash of (task_id, seed)
        # — no `random` dependency needed for reproducibility.
        import hashlib
        remaining.sort(
            key=lambda t: hashlib.sha256(
                f"{adversarial_seed}:{t}".encode()
            ).hexdigest()
        )
        adversarial_picks = remaining[:n_adversarial]

    return (tuple(llm_picked + adversarial_picks), tuple(adversarial_picks))


# ─── Stage 3: execute_probes ────────────────────────────────────────────────


def execute_probes(
    *,
    child_commit_sha: str,
    probe_tasks: Sequence[str],
    runner: HarborRunner,
    adversarial_picks: Sequence[str] = (),
) -> ProbeResults:
    """Stage 3 — invoke Harbor on the K probe tasks via the runner protocol.

    Pure delegation. The runner owns the worktree setup + Harbor
    invocation. If ``probe_tasks`` is empty, we return an empty
    ``ProbeResults`` rather than calling the runner (which would
    likely reject an empty task list).

    ``adversarial_picks`` is the subset of ``probe_tasks`` selected
    adversarially (see ``select_probe_tasks``); copied through to the
    ``ProbeResults`` for the verdict stage. Empty in legacy mode.

    The runner may return either:
    - ``{task: reward}``, OR
    - ``({task: reward}, {task: failure_digest_str})``.
    The two-tuple form lets the gate surface MAGE-style first-failure
    evidence in the verdict prompt and the sidecar.
    """
    if not probe_tasks:
        return ProbeResults(
            probe_tasks=(), rewards={}, adversarial_picks=tuple(adversarial_picks),
        )

    raw = runner.run(
        child_commit_sha=child_commit_sha, probe_tasks=probe_tasks,
    )
    digests: Mapping[str, str] = {}
    rewards_in: Mapping[str, Any]
    if isinstance(raw, tuple) and len(raw) == 2:
        rewards_in, digests = raw
    else:
        rewards_in = raw

    # Defensive copy + key normalization (strings only, in case the
    # runner returned Path-like keys).
    rewards_norm: dict[str, float] = {}
    for t in probe_tasks:
        v = rewards_in.get(t)
        if v is None:
            continue
        try:
            rewards_norm[str(t)] = float(v)
        except (TypeError, ValueError):
            continue

    digests_norm: dict[str, str] = {}
    for t, d in (digests or {}).items():
        if not d:
            continue
        digests_norm[str(t)] = str(d)[:1200]   # cap to bound prompt size

    return ProbeResults(
        probe_tasks=tuple(probe_tasks),
        rewards=rewards_norm,
        failure_digests=digests_norm,
        adversarial_picks=tuple(adversarial_picks),
    )


# ─── Stage 4: verdict ───────────────────────────────────────────────────────


_VERDICT_SCHEMA_HINT = """Reply with EXACTLY one JSON object (no prose outside it):

```
{
  "decision": "EQUIVALENT" | "MODIFIED" | "INCONCLUSIVE",
  "rationale": "<one paragraph attributing the verdict>"
}
```

Decision criteria:
- "EQUIVALENT": ALL probe tasks passed (reward >= 1.0). Behavior preserved.
- "MODIFIED":   one or more probe tasks regressed (reward < 1.0) AND the
                regression is plausibly caused by the diff's modified
                surfaces (per the semantic_analysis).
- "INCONCLUSIVE": probes produced no signal (e.g., all errored), OR
                the regressions cannot be attributed to the diff with
                confidence. Conservative default."""


def verdict(
    *,
    semantic_analysis: SemanticAnalysis,
    probe_results: ProbeResults,
    llm: LLMBackend,
    prompt_template: str,
) -> tuple[str, str]:
    """Stage 4 — LLM labels the change ``EQUIVALENT`` / ``MODIFIED`` /
    ``INCONCLUSIVE`` given the semantic_analysis + probe results.

    Returns ``(decision, rationale)``. Decision is always one of the
    three labels (we degrade to ``INCONCLUSIVE`` on parser failure).

    When ALL probes pass we short-circuit to ``EQUIVALENT`` without
    spending an LLM call — the verdict is trivially clear.

    Failure digests (if any) and adversarial-pick attribution are
    injected into the prompt so the LLM can attribute failures
    concretely (MAGE §4.2.1) and weight adversarial regressions
    differently (§4.1.2).
    """
    if not probe_results.probe_tasks:
        return (
            "INCONCLUSIVE",
            "no probe tasks executed (parent had no solved tasks to probe)",
        )

    # Cheap fast path: every probe passed → trivially EQUIVALENT.
    if probe_results.n_fail == 0:
        return ("EQUIVALENT", f"all {probe_results.n_pass} probe tasks passed")

    adversarial_set = set(probe_results.adversarial_picks)
    per_task_lines: list[str] = []
    for t in probe_results.probe_tasks:
        r = probe_results.rewards.get(t, 0.0)
        tag = " [adversarial]" if t in adversarial_set else ""
        per_task_lines.append(f"- {t}{tag}: reward={r:.2f}")

    # First-failure digest block (MAGE-style waveform window for the
    # regressed probes only — cap total to ~4KB).
    digest_lines: list[str] = []
    total = 0
    for t in probe_results.regressed_tasks:
        d = probe_results.failure_digests.get(t)
        if not d:
            continue
        snippet = d[:600]
        digest_lines.append(f"\n#### {t} — failing trial digest\n```\n{snippet}\n```")
        total += len(snippet)
        if total >= 4000:
            digest_lines.append("\n_(further digests truncated)_")
            break
    failure_digests_block = (
        "\n".join(digest_lines) if digest_lines
        else "_(no failure digests available for this run)_"
    )

    adversarial_block = ""
    if probe_results.n_adversarial_regressed > 0:
        adversarial_block = (
            f"\n\n**Adversarial-regression signal:** "
            f"{probe_results.n_adversarial_regressed} of "
            f"{len(adversarial_set)} adversarial picks regressed. "
            f"Adversarial picks were chosen from tasks NOT named by the "
            f"semantic_analyzer — their regression indicates the diff's "
            f"blast radius is larger than the analyzer claimed. Weight "
            f"these strongly toward MODIFIED."
        )

    prompt = (
        prompt_template
        .replace("{modified_surfaces}", ", ".join(semantic_analysis.modified_surfaces))
        .replace("{risk_level}", semantic_analysis.risk_level)
        .replace("{analysis_rationale}", semantic_analysis.rationale or "_none_")
        .replace("{probe_results_table}", "\n".join(per_task_lines))
        .replace("{n_pass}", str(probe_results.n_pass))
        .replace("{n_fail}", str(probe_results.n_fail))
        .replace("{regressed_tasks}", json.dumps(list(probe_results.regressed_tasks)))
    )
    prompt += "\n\n## First-failure digests\n\n" + failure_digests_block
    prompt += adversarial_block
    prompt += "\n\n" + _VERDICT_SCHEMA_HINT

    try:
        raw = llm.complete(prompt=prompt)
    except Exception as e:  # noqa: BLE001
        # Fail-open (ATELIER_GATE_FAIL_OPEN=1): an LLM/gateway hiccup is INFRA,
        # not evidence of a capability regression. Hard-rejecting on it
        # manufactures false negatives. Degrade to INCONCLUSIVE so the full eval
        # arbitrates. Default OFF keeps the legacy MODIFIED behaviour.
        if _env_flag("ATELIER_GATE_FAIL_OPEN", False):
            _LOG.warning(
                "verdict LLM call failed: %s — FAIL-OPEN → INCONCLUSIVE "
                "(defer to full eval; %d/%d probes regressed)",
                e, probe_results.n_fail, len(probe_results.probe_tasks),
            )
            return (
                "INCONCLUSIVE",
                f"verdict LLM unavailable ({e}); fail-open → INCONCLUSIVE "
                f"(deferring to full eval)",
            )
        _LOG.warning("verdict LLM call failed: %s — treating as MODIFIED (n_fail=%d)",
                     e, probe_results.n_fail)
        return (
            "MODIFIED",
            f"verdict LLM unavailable ({e}); {probe_results.n_fail}/"
            f"{len(probe_results.probe_tasks)} probes regressed → MODIFIED",
        )

    parsed = _extract_json_object(raw)
    if parsed is None:
        _LOG.warning(
            "verdict: could not parse JSON (raw=%r); treating as INCONCLUSIVE",
            raw[:200],
        )
        return ("INCONCLUSIVE", "JSON parse failure on verdict reply")

    decision_raw = str(parsed.get("decision", "")).strip().upper()
    if decision_raw not in {"EQUIVALENT", "MODIFIED", "INCONCLUSIVE"}:
        decision_raw = "INCONCLUSIVE"

    rationale = str(parsed.get("rationale", "")).strip() or "no rationale provided"
    return (decision_raw, rationale)


def verdict_with_consensus(
    *,
    semantic_analysis: SemanticAnalysis,
    probe_results: ProbeResults,
    llm: LLMBackend,
    prompt_template: str,
    n_votes: int = 3,
) -> tuple[str, str, tuple[str, ...]]:
    """Stage 4 with N independent verdict votes aggregated by majority
    (CANDOR pattern, arXiv 2605.18747 §4.1.1 — three Panelists +
    Curator). Returns ``(decision, aggregated_rationale, all_votes)``.

    Aggregation rules:
    - all probes passed → short-circuit to EQUIVALENT (no LLM calls,
      no votes recorded).
    - any vote returns MODIFIED → final decision MODIFIED (the
      conservative "any panelist rejects" rule; a regression caught
      by even one vote is worth investigating).
    - else if EQUIVALENT count ≥ ⌈n_votes/2⌉ → EQUIVALENT.
    - else → INCONCLUSIVE.

    Rationale of the picked decision is whichever vote's rationale
    matched the aggregated decision (first match). All raw votes are
    returned for the sidecar / retrospective audit.

    When ``n_votes <= 1``, falls back to single-call ``verdict()`` for
    backward compatibility.
    """
    if n_votes <= 1:
        d, r = verdict(
            semantic_analysis=semantic_analysis,
            probe_results=probe_results,
            llm=llm,
            prompt_template=prompt_template,
        )
        return (d, r, ())

    if not probe_results.probe_tasks:
        return ("INCONCLUSIVE", "no probe tasks executed", ())
    if probe_results.n_fail == 0:
        return ("EQUIVALENT", f"all {probe_results.n_pass} probe tasks passed", ())

    decisions: list[str] = []
    rationales: list[str] = []
    for i in range(n_votes):
        d, r = verdict(
            semantic_analysis=semantic_analysis,
            probe_results=probe_results,
            llm=llm,
            prompt_template=prompt_template,
        )
        decisions.append(d)
        rationales.append(r)
        _LOG.info("verdict vote %d/%d: %s", i + 1, n_votes, d)

    # Aggregation. Legacy (default): any single panelist labelling MODIFIED
    # forces MODIFIED — maximally conservative, but a lone dissenting/noisy
    # panelist hard-rejects an otherwise-equivalent candidate. With
    # ATELIER_GATE_MAJORITY=1 we require a MAJORITY of panelists for MODIFIED
    # too (symmetric with the EQUIVALENT rule), keeping only a mild safety bias
    # via the tie→INCONCLUSIVE fallthrough (INCONCLUSIVE still gets a full eval).
    mod_count = decisions.count("MODIFIED")
    eq_count = decisions.count("EQUIVALENT")
    majority = (n_votes + 1) // 2
    if _env_flag("ATELIER_GATE_MAJORITY", False):
        if mod_count >= majority:
            agg = "MODIFIED"
        elif eq_count >= majority:
            agg = "EQUIVALENT"
        else:
            agg = "INCONCLUSIVE"
    elif "MODIFIED" in decisions:
        agg = "MODIFIED"
    elif eq_count >= majority:
        agg = "EQUIVALENT"
    else:
        agg = "INCONCLUSIVE"

    # Pick a rationale whose decision matched the aggregate.
    chosen = "no matching rationale"
    for d, r in zip(decisions, rationales):
        if d == agg:
            chosen = r
            break

    summary = (
        f"consensus {decisions.count(agg)}/{n_votes} → {agg}. "
        f"votes={decisions}. {chosen}"
    )
    return (agg, summary, tuple(decisions))


# ─── Stage 5: extension_check ───────────────────────────────────────────────


def extension_check(
    *,
    child_solved: Iterable[str],
    parent_unsolved: Iterable[str],
) -> ExtensionResult:
    """Stage 5 — does the child solve at least one task the parent left
    unsolved?

    Pure set arithmetic. No LLM. This is the second half of the
    monotonic-improvement contract: equivalence preserves the parent's
    solves, extension adds at least one new solve.
    """
    parent_unsolved_set = {str(t) for t in parent_unsolved if t}
    child_solved_set = {str(t) for t in child_solved if t}
    new = tuple(sorted(child_solved_set & parent_unsolved_set))
    return ExtensionResult(
        extension_tasks_solved=new,
        has_extension=len(new) >= 1,
    )


# ─── Aggregate ──────────────────────────────────────────────────────────────


def aggregate_verdict(
    *,
    semantic_analysis: SemanticAnalysis | None,
    probe_results: ProbeResults,
    decision: str,
    rationale: str,
    extension: ExtensionResult | None = None,
    extension_required: bool = True,
    parent_commit_sha: str | None = None,
    parent_solved_count: int | None = None,
    consensus_votes: tuple[str, ...] = (),
) -> EquivalenceVerdict:
    """Combine the five stages into the single ``EquivalenceVerdict`` the
    pipeline reads.

    Accept iff:
      - decision == EQUIVALENT, AND
      - extension is not required, OR extension.has_extension is True.

    ``extension`` may be None when the caller cannot compute it yet
    (e.g., the equivalence gate runs before final-eval and we don't
    know child_solved). In that case the gate accepts on EQUIVALENT
    alone and the extension contract is enforced at the post-final-eval
    stage instead.

    New (post-survey enhancements):
    - ``parent_commit_sha`` + ``parent_solved_count`` populate the
      VerificationScope and ChangeContract sidecar fields so the
      verdict carries its own evidence bundle (arXiv 2605.18747
      §§5.2.2-5.2.3).
    - ``consensus_votes`` records the raw votes from
      ``verdict_with_consensus`` for auditability.
    """
    # Regression-tolerance: optionally downgrade a small/net-positive MODIFIED
    # to INCONCLUSIVE so the full eval (ground truth) arbitrates instead of the
    # gate hard-killing it. No-op unless ATELIER_GATE_REGRESSION_TOL>0.
    decision, _tol_note = _apply_gate_tolerance(decision, probe_results, extension)
    if _tol_note:
        rationale = f"{rationale} | {_tol_note}"

    if decision == "EQUIVALENT":
        if not extension_required or extension is None:
            accept = True
        else:
            accept = extension.has_extension
    else:
        accept = False

    # ─── §5.2.2 — verification scope ───────────────────────────────
    scope: VerificationScope | None = None
    if semantic_analysis is not None:
        # Conservatively mark all named surfaces as "verified" only when
        # at least one probe ran. The fine-grained task→surface mapping
        # is left for a future enhancement; today we surface the gap
        # honestly: K/N coverage, untested assumption recorded.
        verified = (
            semantic_analysis.modified_surfaces
            if probe_results.probe_tasks else ()
        )
        unverified = tuple(
            s for s in semantic_analysis.modified_surfaces
            if s not in verified
        )
        if parent_solved_count and parent_solved_count > 0:
            coverage = min(
                1.0, len(probe_results.probe_tasks) / parent_solved_count,
            )
        else:
            coverage = 0.0
        scope = VerificationScope(
            verified_surfaces=verified,
            unverified_surfaces=unverified,
            probe_coverage=coverage,
            untested_assumption=(
                f"behavior on the {parent_solved_count - len(probe_results.probe_tasks)} "
                f"unprobed parent-solved tasks is preserved iff it is preserved "
                f"on the {len(probe_results.probe_tasks)} probed tasks"
                if parent_solved_count
                and parent_solved_count > len(probe_results.probe_tasks)
                else None
            ),
        )

    # ─── §5.2.3 — change contract ─────────────────────────────────
    contract: ChangeContract | None = None
    if probe_results.probe_tasks and semantic_analysis is not None:
        if decision == "EQUIVALENT":
            invariants = (
                "parent.solved ⊆ child.solved on K probe tasks",
            )
            # Falsifier for an EQUIVALENT verdict: the same K probes,
            # any one of which regressing would have flipped us to
            # MODIFIED.
            falsifier = probe_results.probe_tasks
        else:
            invariants = ()
            falsifier = probe_results.regressed_tasks

        rollback = (
            f"git checkout {parent_commit_sha}"
            if parent_commit_sha else "git checkout <parent_commit_sha>"
        )
        contract = ChangeContract(
            invariants_preserved=invariants,
            falsifier=falsifier,
            rollback_action=rollback,
            modified_component="+".join(semantic_analysis.modified_surfaces)
                                or "unknown",
        )

    return EquivalenceVerdict(
        decision=decision,
        k_picked_tasks=probe_results.probe_tasks,
        per_task_results=dict(probe_results.rewards),
        extension_tasks_solved=(extension.extension_tasks_solved if extension else ()),
        n_regressions=probe_results.n_fail,
        semantic_analysis=semantic_analysis,
        verdict_rationale=rationale,
        accept_for_final_eval=accept,
        extension_required=extension_required,
        verification_scope=scope,
        change_contract=contract,
        consensus_votes=consensus_votes,
        failure_digests=dict(probe_results.failure_digests),
    )


# ─── Sidecar serialization ──────────────────────────────────────────────────


def verdict_to_dict(v: EquivalenceVerdict) -> dict[str, Any]:
    """JSON-safe representation for the equivalence sidecar.

    Schema v2 (post-survey enhancements) adds:
    - ``verification_scope``: what was/wasn't verified + confidence
    - ``change_contract``: invariants, falsifier, rollback
    - ``consensus_votes``: raw votes from multi-vote verdict
    - ``failure_digests``: first-failure attribution for regressed
                           probes (MAGE-style)
    """
    out: dict[str, Any] = {
        "schema_version": 2,
        "decision": v.decision,
        "accept_for_final_eval": v.accept_for_final_eval,
        "n_regressions": v.n_regressions,
        "k_picked_tasks": list(v.k_picked_tasks),
        "per_task_results": dict(v.per_task_results),
        "extension_tasks_solved": list(v.extension_tasks_solved),
        "extension_required": v.extension_required,
        "verdict_rationale": v.verdict_rationale,
        "consensus_votes": list(v.consensus_votes),
        "failure_digests": dict(v.failure_digests),
    }
    if v.semantic_analysis is not None:
        sa = v.semantic_analysis
        out["semantic_analysis"] = {
            "modified_surfaces": list(sa.modified_surfaces),
            "risk_level": sa.risk_level,
            "rationale": sa.rationale,
            # raw_response intentionally omitted (often >10 KB)
        }
    if v.verification_scope is not None:
        vs = v.verification_scope
        out["verification_scope"] = {
            "verified_surfaces": list(vs.verified_surfaces),
            "unverified_surfaces": list(vs.unverified_surfaces),
            "probe_coverage": vs.probe_coverage,
            "untested_assumption": vs.untested_assumption,
        }
    if v.change_contract is not None:
        cc = v.change_contract
        out["change_contract"] = {
            "invariants_preserved": list(cc.invariants_preserved),
            "falsifier": list(cc.falsifier),
            "rollback_action": cc.rollback_action,
            "modified_component": cc.modified_component,
        }
    return out


def write_verdict_sidecar(
    *,
    reports_root: Path,
    campaign: str,
    child_node_id: str,
    verdict: EquivalenceVerdict,
) -> Path:
    """Persist the verdict JSON next to the campaign's other Atelier
    sidecars. Returns the written path."""
    out_dir = Path(reports_root) / campaign / "atelier" / "equivalence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{child_node_id}.equivalence.json"
    out_path.write_text(
        json.dumps(verdict_to_dict(verdict), indent=2, default=str),
        encoding="utf-8",
    )
    return out_path


# ─── Helpers ────────────────────────────────────────────────────────────────


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """Return the first JSON object found in ``raw``, or None.

    Tries (in order): a ```json fenced block, then a greedy ``{...}``
    match. Both fall back to None on parse failure so callers can
    handle the degraded case.
    """
    if not raw:
        return None

    m = _JSON_FENCE_RE.search(raw)
    candidates: list[str] = []
    if m:
        candidates.append(m.group(1))
    m2 = _FIRST_OBJ_RE.search(raw)
    if m2:
        candidates.append(m2.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ─── OpenAI-compatible backend (production default) ────────────────────────


@dataclass
class OpenAIChatConfig:
    """Free-text chat-completion config (no logprobs).

    Lighter than ``verifier_backend.OpenAIVerifierConfig`` because the
    matchfix gate's three LLM calls only need plain text (the JSON we
    parse ourselves), not per-token logprob distributions.
    """

    model: str
    api_key: str
    base_url: str | None = None
    # ``None`` => omit the temperature param entirely. Reasoning models
    # (e.g. gpt-5.5 on LLM Gateway Express) reject any non-default
    # temperature with a 400, so the express path passes None.
    temperature: float | None = 0.1
    max_tokens: int = 1024
    request_timeout: float = 90.0


class OpenAIChatBackend:
    """Plain chat-completion ``LLMBackend`` implementation.

    Lazy-imports openai so the module can be imported without that dep
    available (matching ``verifier_backend``'s pattern).
    """

    def __init__(self, config: OpenAIChatConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "atelier.matchfix_gate.OpenAIChatBackend requires "
                    "openai. Install via `uv sync --extra atelier`."
                ) from e
            kwargs: dict[str, Any] = {
                "api_key": self.config.api_key,
                "timeout": self.config.request_timeout,
            }
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            # The SFR Gateway authenticates via `x-api-key:` header,
            # not the OpenAI-standard `Authorization: Bearer ...`.
            # Without this, every request returns 401 Not Authenticated
            # (the failure mode that silently disabled all v8 LLM calls).
            # Both headers are sent; the gateway picks x-api-key and
            # ignores the bearer.
            if self.config.base_url and "gateway.salesforce" in self.config.base_url:
                kwargs["default_headers"] = {"x-api-key": self.config.api_key}
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(self, *, prompt: str, system: str | None = None) -> str:
        client = self._get_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # The gpt-5.x models on the SFR Gateway require
        # `max_completion_tokens` rather than the older `max_tokens`.
        # We pass max_completion_tokens unconditionally — for older
        # endpoints that don't accept it, switch to `max_tokens` via
        # extra_create_kwargs override.
        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_completion_tokens": self.config.max_tokens,
        }
        # Omit temperature entirely when None — reasoning models reject any
        # non-default value with a 400.
        if self.config.temperature is not None:
            create_kwargs["temperature"] = self.config.temperature
        response = client.chat.completions.create(**create_kwargs)
        if not response.choices:
            return ""
        return (response.choices[0].message.content or "").strip()


def chat_backend_from_credentials(
    *,
    model: str,
    provider: str = "sfr_gateway",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    request_timeout: float = 90.0,
) -> OpenAIChatBackend:
    """Build an ``OpenAIChatBackend`` from the project's
    ``monet_eval.core.credentials`` (matches ``verifier_backend``'s
    pattern).
    """
    # coding-bench has no ``monet_eval.core.credentials``; the campaign
    # launch exports the provider credentials into the process environment
    # (the Express key + local-proxy URL), so resolve straight from
    # ``os.environ`` instead of the monet credentials registry.
    import os

    env = dict(os.environ)
    base_url = env.get("OPENAI_BASE_URL")
    if provider == "sfr_gateway":
        # SFR Gateway is a real network endpoint reachable from BOTH the
        # control-plane/launch host (where this gate reasoner runs) AND the
        # cluster nodes, so it is the recommended gate provider for a
        # cluster (xrlenv) run — unlike the Express *local* proxy, which is
        # a host-loopback tunnel. Accept the SAME creds the monet
        # agent-under-test uses on the SFR route (``X_API_KEY`` /
        # ``SFR_GATEWAY_OPENAI_URL``) as fallbacks, so ONE set of SFR creds
        # drives both the agent and the gate; the canonical ``OPENAI_*``
        # names still win when set.
        api_key = env.get("OPENAI_GATEWAY_API_KEY") or env.get("X_API_KEY")
        if not api_key:
            raise KeyError(
                "sfr_gateway gate reasoner requires OPENAI_GATEWAY_API_KEY "
                "(or X_API_KEY)"
            )
        if not base_url:
            base_url = env.get("SFR_GATEWAY_OPENAI_URL")
    elif provider == "openai":
        api_key = env["OPENAI_API_KEY"]
    elif provider == "llm_gateway_express_local_proxy":
        # The gate reasoner shares the laptop->Express reverse tunnel with
        # the evaluated agent. The express local proxy only routes
        # ``/chat/completions`` and ``/responses`` (NO ``/v1`` prefix), and
        # the openai client appends ``/chat/completions`` to ``base_url`` —
        # so point base_url at the proxy ROOT with the trailing slash
        # stripped. Auth is a standard ``Authorization: Bearer`` (the proxy
        # forwards it verbatim), so no x-api-key header is needed.
        api_key = env["LLM_GATEWAY_EXPRESS_API_KEY"]
        proxy = (
            env.get("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL")
            or "http://127.0.0.1:18080/"
        )
        base_url = proxy.rstrip("/")
        # gpt-5.5 (and the claude reasoning models) on Express reject any
        # non-default temperature -> omit it for this provider.
        temperature = None
    else:
        raise ValueError(
            "matchfix_gate chat backend supports openai / sfr_gateway / "
            f"llm_gateway_express_local_proxy, got {provider!r}"
        )
    return OpenAIChatBackend(
        OpenAIChatConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )
    )


__all__ = [
    # protocols
    "LLMBackend",
    "HarborRunner",
    # dataclasses
    "SemanticAnalysis",
    "ProbeResults",
    "ExtensionResult",
    "VerificationScope",
    "ChangeContract",
    "EquivalenceVerdict",
    # stages
    "analyze_diff",
    "select_probe_tasks",
    "execute_probes",
    "verdict",
    "verdict_with_consensus",
    "extension_check",
    "aggregate_verdict",
    # sidecar
    "verdict_to_dict",
    "write_verdict_sidecar",
    # production backend
    "OpenAIChatConfig",
    "OpenAIChatBackend",
    "chat_backend_from_credentials",
]
