"""v9 — Reasoned verdict (a "trace-analyzer-styled" upgrade of the accept slot).

The legacy accept decision is coarse: a scalar fitness blend
((1-alpha)*pass_rate + alpha*verifier) plus net-gain/threshold tests, with the
qualitative *reasons* thrown away. This module replaces that with ONE reasoned
agent that reads the **structured, evidence-cited** picture as text —

  - per-task gain: which claimed tasks went which way (before -> after rate),
  - the diff summary: what the change actually does,
  - the trace_qc fault-localization gradient that motivated the change,
  - (optional) regression-probe results on diff-adjacent tasks,

and returns a reasoned ``ACCEPT`` / ``REJECT`` / ``ARCHIVE`` (stepping stone) with
a confidence and a short rationale.

Design rules (so we don't recreate the heaviness):
  * **Env-gated** — ``DARWINX_GATE_REASONED_VERDICT=1`` turns it on; off = legacy path.
  * **One call per node** — cheap relative to the eval; not per-chunk/per-task.
  * **Grounded** — the prompt forbids inventing evidence; the agent must cite the
    provided deltas/digests. It's a *judgment over given evidence*, not a probe.
  * **Fail-open** — any error (no client, bad parse) returns ``None`` so the caller
    falls back to its scalar decision. Never blocks the loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

from . import null_calibration

log = logging.getLogger("self_evolve.reasoned_verdict")

_TRUTHY = {"1", "true", "yes", "on"}


def reasoned_verdict_on() -> bool:
    return os.environ.get("DARWINX_GATE_REASONED_VERDICT", "0").strip().lower() in _TRUTHY


class _LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class Verdict:
    # PROMOTE = make this the active tip (clean improvement).
    # ARCHIVE = real stepping stone (helped something, regressed something) —
    #           keep + build on, don't promote.
    # REJECT  = no real gain / pure regression — don't keep this iteration's edit;
    #           the NODE is still archived (never dropped) so its lesson survives.
    # POISON  = cheat / eval-tampering — never promote, never build on; archived
    #           only as an anti-pattern lesson.
    decision: str          # "PROMOTE" | "ARCHIVE" | "REJECT" | "POISON"
    confidence: float      # 0..1
    rationale: str
    # --- gate-as-activation: a verdict ALWAYS emits a forward signal, even on
    # REJECT/POISON. The gate is an activation function (it can decline to
    # promote a bad value) that still passes a *gradient* (lesson) forward.
    lesson: str = ""           # transferable textual gradient from this attempt
    next_directive: str = ""   # the single most useful thing to try next
    surface: str = ""          # "code" | "skill" | "mixed" — which surface was edited

    @property
    def accept(self) -> bool:
        """PROMOTE — keep this iteration's edit and make it the active tip."""
        return self.decision == "PROMOTE"

    @property
    def archive(self) -> bool:
        """ARCHIVE — keep as a stepping stone (build on it) without promoting."""
        return self.decision == "ARCHIVE"

    @property
    def poisoned(self) -> bool:
        """POISON — cheat/tamper; never promote or build on; archive as anti-pattern."""
        return self.decision == "POISON"


_SYSTEM = """\
You are the GATE (an activation function) for a self-evolution loop that edits a
coding agent's source to fix tasks it fails. You decide the FATE of ONE candidate
edit by REASONING over context — not by comparing a number to a threshold.

Decide one of:
- PROMOTE: a genuine, generalizable improvement — make it the active tip. The
  mechanism (per the diff + fault analysis) should plausibly generalize, not look
  like overfitting to one task's answer, and there is no convincing regression.
- ARCHIVE: a real stepping stone — it helped something genuine but also regressed
  something (or is promising-but-unproven). Worth keeping and building on; not
  ready to be the tip. (Nothing is thrown away — archived edits + their lessons
  feed future attempts.)
- REJECT: no real gain — the apparent gain is most likely sampling luck/noise, or
  it regresses more than it helps, or it's a risky rewrite of shared code. This
  iteration's edit is not kept, BUT its lesson is still recorded for the future.
- POISON: the edit games the verifier (hardcoded answers, eval/test tampering,
  task-name special-casing). Never promote or build on it; record the anti-pattern.

HOW TO WEIGH THE EVIDENCE (critical):
- The pass-rate NUMBERS are NOISY context, not a verdict. A rate over very few
  samples (e.g. k=2) carries little information: 0 -> 0.5 is one lucky run, not a
  win; 1.0 -> 0.5 may be jitter, not a regression. Do NOT treat any single number
  as decisive or apply a fixed cutoff.
- Decide by COMBINING the numbers with: the fault-analysis gradient (does the
  diff actually address the diagnosed failure?), the diff itself (is the mechanism
  sound and additive, or a fragile shared-core rewrite?), and the COLLECTIVE
  KNOWLEDGE below (have similar edits/surfaces repeatedly helped or repeatedly
  regressed across this campaign?). A small numeric gain that the gradient +
  history corroborate beats a large one that nothing explains.
- Be MORE confident PROMOTE when prior collective knowledge shows this kind of
  change worked; lean ARCHIVE/REJECT when history shows this surface/pattern keeps
  regressing.

- ADDITIVE-SKILL BIAS-TO-PROMOTE (the gate must ENABLE progress, not just gate it):
  an additive SKILL cannot regress other tasks, so its downside is bounded. If the
  candidate is a PURE SKILL edit (surface == "skill"; NO shared-code/engine
    files touched), shows a measured pass-gain on a previously-failing claimed
  task (before ~0 -> after > 0), and has NO observed regression (no failed guards,
  no regressed existing task), then PROMOTE it — even when k is small and the gain
  is not yet corroborated by fault analysis. RISK-CLASS CARVE-OUT (critical): this bias applies ONLY to pure-skill edits. A MIXED edit (skill + shared code) or a CODE edit does NOT qualify — its code portion has UNBOUNDED blast radius and can regress tasks OUTSIDE the mini-eval guard (exactly how a node scoring 1.0 on mini-eval later failed full avg@k: 7 fixed, 14 regressed). For any edit touching shared code (surface == "code" or "mixed"), require the FULL generalizability bar + widened guard; do NOT PROMOTE on a small-k claimed win alone — ROUTED-CODE EXCEPTION: a shared-code edit that is properly ROUTED — guarded on the task's capability-cluster, with a BYTE-IDENTICAL else-branch and fail-open on classification uncertainty — has BOUNDED blast radius (it cannot change behavior for tasks outside its target cluster) and may take the SAME cheap PROMOTE path as an additive skill, PROVIDED the guard is verified not to fire outside the target cluster. An UNROUTED shared-code edit does NOT qualify. Otherwise, prefer retargeting the fix as a pure additive skill. The RIGOROUS gate is downstream: the
  deferred avg@k confirmation + the preservation probes re-test every promoted node
  and will demote a dud. The campaign cannot improve if every unproven-but-harmless
  additive win is archived, so for bounded-downside additive wins, prefer PROMOTE
  over ARCHIVE. Reserve ARCHIVE for skills that ALSO regressed something or had ZERO
  measured effect; reserve REJECT for net-regressions, noise-only after-rates with no
  before-failure, or risky shared-code rewrites.

- PRUNE CANDIDATES (subtraction) — judged by a DIFFERENT rule. When the evidence
  block below is marked as a PRUNE candidate, the edit's purpose is to REMOVE an
  accumulated modification, not to add capability. Do not apply the "zero measured
  effect -> ARCHIVE" rule to it: for a prune, preserving every task while removing
  code IS the intended, successful outcome, because a smaller agent is cheaper and
  easier to improve. Judge it as follows:
    * PROMOTE if no task regressed and the removal is real (the diff net-removes
      code). Unchanged pass-rates are the SUCCESS condition here, not evidence of
      a no-op. PROMOTE with extra confidence if the score also went UP — that means
      the removed modification was actively harmful.
    * ARCHIVE if it removed something real but also regressed a task, so the
      removed code was partly load-bearing and the lesson is worth keeping.
    * REJECT if it regressed tasks without removing anything meaningful, or if it
      smuggled in new capability instead of subtracting (a prune's diff must be
      dominated by deletions), or if the removal is trivially cosmetic.
    * POISON if it deleted or weakened a test, an assertion, a verifier, or an
      evidence/audit check. Removing a check is never a valid prune — that is eval
      tampering and the strongest possible reason to poison a node.
  Weigh the load-bearing question explicitly: the danger of subtraction is removing
  something that only matters on tasks outside the screened slice. If the rationale
  for deadness/redundancy is weak and unsupported by the trajectories, prefer
  ARCHIVE over PROMOTE.

- CONSOLIDATE CANDIDATES (rewrite for simplicity) — judged by a THIRD rule.
  When the evidence block is marked as a CONSOLIDATE candidate, the edit's purpose
  is to hold capability at lower complexity: fold accumulated special cases into
  one general mechanism, or rewrite a pre-evolve component the agent has outgrown.
  This is the ONLY variant permitted to rewrite base/pre-evolve code, and it is
  expected to — do not hold that against it. Judge it as follows:
    * PROMOTE if no task regressed AND the complexity evidence shows a real
      reduction (fewer total lines, or materially fewer branches at similar
      lines — the latter is the stronger signal, since collapsing three special
      cases into one general path is exactly the intended move). Unchanged
      pass-rates are the SUCCESS condition here. PROMOTE with extra confidence if
      the score also rose: the accumulated special cases were getting in the way.
    * ARCHIVE if it simplified something real but regressed a task, or if
      capability held but the complexity reduction is negligible or cosmetic
      (renames, reflowing, moving code between files).
    * REJECT if it regressed tasks, or if complexity went UP — a rewrite that is
      bigger AND no better is the failure mode this variant exists to avoid.
    * POISON if it deleted or weakened a test, an assertion, a verifier, or an
      evidence check, or if it removed a capability contract (e.g. how the agent
      submits, or its step/cost limits) rather than re-expressing it. Simplifying
      by deleting the thing being measured is never a valid consolidation.
  A rewrite's danger is losing behavior that only shows up outside the screened
  slice. Where the justification for equivalence is asserted rather than argued
  from the trajectories, prefer ARCHIVE over PROMOTE.

The edit targets ONE surface: CODE (monet's shared engine in src/ — high
regression risk) or a SKILL (an additive procedure — cannot regress other tasks).
If a CODE edit regressed, the usual next move is to retarget the fix as an additive
SKILL.

You ALWAYS emit `lesson` and `next_directive`, for EVERY decision — even REJECT/
POISON. The lesson must be transferable (usable by a proposal on a different task);
the directive must be one concrete, actionable next step (it may say to switch
surface, e.g. "add a SKILL for X instead of editing the loop").

Respond with ONLY JSON:
{"decision":"PROMOTE|ARCHIVE|REJECT|POISON","confidence":0.0-1.0,"rationale":"<=300 chars",
 "lesson":"<=300 chars: the transferable lesson from THIS attempt (textual gradient)",
 "next_directive":"<=200 chars: the single most useful thing to try next"}"""


# ── agent-profile view of the rubric ────────────────────────────────────────
# The rubric above is written for monet, which has TWO surfaces: a shared JS
# engine and an additive skill registry. Its bias-to-PROMOTE is granted only to
# pure-SKILL edits, and its standing advice on a risky edit is "retarget it as a
# skill".
#
# mini-swe-agent has no skill registry. Every possible edit classifies as
# "code", so under the monet rubric every mini candidate faces the strictest bar
# AND is told to retarget onto a surface that does not exist -- the same shape
# of failure the Jul 30 audit found, where a rule written for one architecture
# became the binding constraint and rejected 123 candidates for one violation.
#
# So for mini we restate the two surface-dependent passages in terms of the
# surfaces mini actually has. Everything else -- evidence standards, prune
# rules, POISON, output schema -- is shared and untouched.
_MINI_BIAS = """\
- BOUNDED-DOWNSIDE BIAS-TO-PROMOTE (the gate must ENABLE progress, not just gate it):
  this agent is ~370 lines -- a small agent loop plus its prompt config -- and has
  NO additive skill registry, so EVERY edit classifies as "code". Do not hold that
  classification against a candidate and never advise retargeting the fix as a
  skill: there is no such surface here. Judge blast radius from the DIFF instead.
  A PROMPT-CONFIG edit (the agent's instruction templates) cannot change control
  flow and is reverted by a one-file rollback, so its downside is bounded and
  legible. If such a candidate shows a measured pass-gain on a previously-failing
  claimed task (before ~0 -> after > 0) with NO observed regression, PROMOTE it,
  even at small k -- deferred avg@k confirmation re-tests it and will demote a dud.
  Hold LOOP edits (control flow, submission handling, limits) to the fuller bar:
  they can regress tasks outside the mini-eval guard. Reserve ARCHIVE for edits
  that regressed something or had ZERO measured effect; reserve REJECT for
  net-regressions and for noise-only after-rates with no before-failure."""

_MINI_SURFACES = """\
The edit targets ONE surface: the PROMPT CONFIG (instruction templates -- bounded,
one-file rollback) or the LOOP (control flow, observation handling, submission and
limit enforcement -- higher regression risk). If a LOOP edit regressed, the usual
next move is to express the same intent in the PROMPT CONFIG instead."""


def _agent_profile_system() -> str:
    """The rubric, restated for the agent actually under evolution."""
    if os.environ.get("DARWINX_GATE_VERDICT_AGENT_PROFILE", "").strip().lower() != "mini":
        return _SYSTEM
    text = _SYSTEM
    bias_start = text.find("- ADDITIVE-SKILL BIAS-TO-PROMOTE")
    bias_end = text.find("- PRUNE CANDIDATES", bias_start) if bias_start != -1 else -1
    if bias_start != -1 and bias_end != -1:
        text = text[:bias_start] + _MINI_BIAS + "\n\n" + text[bias_end:]
    surf_start = text.find("The edit targets ONE surface:")
    surf_end = text.find("You ALWAYS emit", surf_start) if surf_start != -1 else -1
    if surf_start != -1 and surf_end != -1:
        text = text[:surf_start] + _MINI_SURFACES + "\n\n" + text[surf_end:]
    return text


def render_evidence(
    *,
    task_deltas: list[dict[str, Any]],
    diff_summary: str | None,
    gradient_digest: str | None,
    probe_results: str | None = None,
    k_samples: int | None = None,
    surface: str | None = None,
    collective_knowledge: str | None = None,
    prune: bool = False,
    consolidate: bool = False,
    complexity: str | None = None,
    panel_null: "null_calibration.PanelNull | None" = None,
) -> str:
    """Assemble the textual evidence block the judge reasons over.

    ``task_deltas``: list of {task, before_rate, after_rate} (rates in [0,1]).
    ``surface``: which surface the edit touched ("code" | "skill" | "mixed").
    ``collective_knowledge``: campaign-wide digest of how prior edits fared
    (what worked / what regressed across ALL nodes) — the aggregated history the
    GATE weighs alongside this node's noisy numbers.
    ``prune``: this candidate REMOVES an accumulated modification. Flagged first
    and prominently because it inverts the success condition — unchanged
    pass-rates mean success, not a no-op — and the judge must not fall back on the
    additive reading.
    ``consolidate``: this candidate REWRITES part of the harness — prior edits or
    the pre-evolve base — to hold capability at lower complexity. Like a prune it
    inverts the success condition, but unlike a prune its diff is NOT expected to
    be deletion-dominated: folding three special cases into one general mechanism
    is the intended shape, and judging it by deletion count would reject it.
    ``complexity``: measured size/branching of the harness before and after; the
    second axis a consolidation is judged on.
    ``panel_null``: measured behaviour of this panel with the harness UNCHANGED.
    When supplied, each delta is labelled with that task's stability and the
    block states what the panel can resolve. Defaults to ``None``, which
    reproduces the previous output byte for byte. It is passed rather than
    loaded here so a caller can decide; ``null_calibration.load_from_env`` is
    the usual source.
    """
    lines: list[str] = []
    if consolidate:
        lines.append(
            "### THIS IS A **CONSOLIDATE** CANDIDATE (rewrite) — judge it by the "
            "CONSOLIDATE rule, not the additive rule and not the prune rule.\n"
            "Its goal is to hold capability while making the harness SIMPLER: "
            "replacing accumulated special cases with one general mechanism, or "
            "rewriting a pre-evolve component that has been outgrown. UNCHANGED "
            "pass-rates plus a real complexity reduction are the SUCCESS condition. "
            "Do NOT require the diff to be deletion-dominated — a rewrite legitimately "
            "adds lines in one place while removing more in another. Do NOT treat "
            "editing pre-evolve/base code as suspicious here; it is the point. Only "
            "regressions, an absent complexity reduction, or a weakened test count "
            "against it."
        )
    if complexity:
        lines.append(complexity)
    if prune:
        lines.append(
            "### THIS IS A **PRUNE** CANDIDATE (subtraction) — judge it by the PRUNE "
            "rule, not the additive rule.\n"
            "Its goal is to REMOVE an accumulated modification (a dead/redundant/"
            "harmful skill or guarded branch), not to add capability. UNCHANGED "
            "pass-rates below are the SUCCESS condition: behavior preserved + code "
            "removed = PROMOTE. A score INCREASE means the removed modification was "
            "actively harmful. Only regressions, a diff that fails to remove "
            "anything, or a deleted test/verifier count against it."
        )
    if surface:
        lines.append(f"Edited surface: **{surface.upper()}** "
                     f"(code = shared engine, regression-prone; skill = additive, safe).")
    if k_samples:
        lines.append(
            f"Each rate is over k={k_samples} samples — TREAT AS NOISY: "
            f"at k={k_samples} a single sample moves the rate by "
            f"~{1.0 / max(1, k_samples):.2f}, so small before->after moves are "
            f"weak evidence. Weigh them with the diff, fault analysis, and "
            f"collective knowledge below; do not threshold on the number."
        )
    header = null_calibration.evidence_header(panel_null, k_samples)
    if header:
        lines.append(header)
    lines.append("## Claimed-task gain (before -> after pass-rate) [noisy — context, not a verdict]")
    if task_deltas:
        for d in task_deltas:
            lines.append(null_calibration.annotate_delta(
                d.get("task", "?"), d.get("before_rate"), d.get("after_rate"),
                panel_null,
            ))
        if panel_null is not None:
            up, down, ignored = null_calibration.net_stable_delta(task_deltas, panel_null)
            lines.append(
                f"\nRestricted to tasks the null calls STABLE: {up} improved, "
                f"{down} regressed, {ignored} excluded as flaky/unmeasured. "
                f"Movement on the excluded tasks is not attributable to this edit."
            )
    else:
        lines.append("- (none recorded)")
    lines.append("\n## What the diff does")
    lines.append((diff_summary or "(no diff summary available)").strip()[:2000])
    lines.append("\n## Fault analysis that motivated the change (trace_qc gradient)")
    lines.append((gradient_digest or "(no fault analysis available)").strip()[:2500])
    if probe_results:
        lines.append("\n## Regression probes on diff-adjacent tasks")
        lines.append(probe_results.strip()[:1500])
    if collective_knowledge and collective_knowledge.strip():
        lines.append("\n## Collective knowledge — how SIMILAR edits fared across this campaign")
        lines.append("(Aggregated across ALL prior nodes incl. failed/regressed. Use this "
                     "as your prior: repeatedly-regressing patterns -> lean ARCHIVE/REJECT; "
                     "repeatedly-winning patterns -> lean PROMOTE.)")
        lines.append(collective_knowledge.strip()[:3000])
    return "\n".join(lines)


def _parse(raw: str) -> Verdict | None:
    if not raw:
        return None
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        obj = json.loads(text[s : e + 1])
    except json.JSONDecodeError:
        return None
    dec = str(obj.get("decision", "")).strip().upper()
    # Normalize vocabularies. New: PROMOTE/ARCHIVE/REJECT/POISON. Legacy ACCEPT
    # maps to PROMOTE; CHEAT/TAMPER map to POISON. (REJECT keeps its meaning:
    # don't keep this iteration — the node is still archived downstream, never
    # dropped.)
    _NORMALIZE = {
        "PROMOTE": "PROMOTE", "ACCEPT": "PROMOTE", "KEEP": "PROMOTE",
        "ARCHIVE": "ARCHIVE", "STEPPING_STONE": "ARCHIVE",
        "REJECT": "REJECT",
        "POISON": "POISON", "CHEAT": "POISON", "TAMPER": "POISON",
    }
    dec = _NORMALIZE.get(dec)
    if dec is None:
        return None
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return Verdict(decision=dec, confidence=max(0.0, min(1.0, conf)),
                   rationale=str(obj.get("rationale", ""))[:400],
                   lesson=str(obj.get("lesson", ""))[:400],
                   next_directive=str(obj.get("next_directive", ""))[:300])


def decide(evidence: str, llm: _LLM | None = None, *, surface: str = "") -> Verdict | None:
    """Run the reasoned verdict. Returns None on any failure (caller falls back).

    ``surface`` (code/skill/mixed) is recorded on the returned Verdict so the
    caller can keep code and skill improvements separately attributable.
    """
    if not reasoned_verdict_on():
        return None
    try:
        client = llm
        if client is None:
            from dx_trace.llm import OpenAIClient  # Express-aware client
            model = os.environ.get("DARWINX_GATE_REASONED_VERDICT_MODEL") or None
            client = OpenAIClient(model=model)
        raw = client.complete(_agent_profile_system(), evidence)
        v = _parse(raw)
        if v is None:
            log.warning("reasoned_verdict: unparseable reply; falling back")
        elif surface and not v.surface:
            import dataclasses
            v = dataclasses.replace(v, surface=surface)
        return v
    except Exception as exc:  # noqa: BLE001 — never block the loop
        log.warning("reasoned_verdict failed (%s); falling back to scalar decision", exc)
        return None


__all__ = ["reasoned_verdict_on", "Verdict", "render_evidence", "decide"]
