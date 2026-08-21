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
  * **Env-gated** — ``ATELIER_REASONED_VERDICT=1`` turns it on; off = legacy path.
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

log = logging.getLogger("self_evolve.reasoned_verdict")

_TRUTHY = {"1", "true", "yes", "on"}


def reasoned_verdict_on() -> bool:
    return os.environ.get("ATELIER_REASONED_VERDICT", "0").strip().lower() in _TRUTHY


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


def render_evidence(
    *,
    task_deltas: list[dict[str, Any]],
    diff_summary: str | None,
    gradient_digest: str | None,
    probe_results: str | None = None,
    k_samples: int | None = None,
    surface: str | None = None,
    collective_knowledge: str | None = None,
) -> str:
    """Assemble the textual evidence block the judge reasons over.

    ``task_deltas``: list of {task, before_rate, after_rate} (rates in [0,1]).
    ``surface``: which surface the edit touched ("code" | "skill" | "mixed").
    ``collective_knowledge``: campaign-wide digest of how prior edits fared
    (what worked / what regressed across ALL nodes) — the aggregated history the
    GATE weighs alongside this node's noisy numbers.
    """
    lines: list[str] = []
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
    lines.append("## Claimed-task gain (before -> after pass-rate) [noisy — context, not a verdict]")
    if task_deltas:
        for d in task_deltas:
            b = d.get("before_rate"); a = d.get("after_rate"); t = d.get("task", "?")
            tag = ""
            try:
                if a is not None and b is not None:
                    tag = " [up]" if a > b else (" [down]" if a < b else " [unchanged]")
            except TypeError:
                pass
            lines.append(f"- {t}: {b} -> {a}{tag}")
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
            from trace_analyzer.llm import OpenAIClient  # Express-aware client
            model = os.environ.get("ATELIER_REASONED_VERDICT_MODEL") or None
            client = OpenAIClient(model=model)
        raw = client.complete(_SYSTEM, evidence)
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
