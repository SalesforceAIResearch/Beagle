"""Unified preserve-and-extend contract + check for the WHOLE pipeline.

Self-evolution here is a preservation-constrained monotone ratchet: every
candidate — a mutation child, a pairwise merge, or an N-way merge — must
PRESERVE the lineage's known-solved set and EXTEND it (solve >=1 new task).

Historically this discipline lived only in the single-branch MUTATION path (as a
proposer-side contract), while the MERGE path used a weaker parent-wins-only
post-hoc veto — which is why "additive" skill bundles were observed to regress
tasks outside the two parents. This module unifies both surfaces:

  - build_contract(solved, unsolved, kind): the GENERATIVE objective, fed to the
    proposer AND to the merge / N-way combiner.
  - preserved(candidate_solved, invariant): the CHECK, applied at every eval.
  - union_solved(*nodes): union of nodes' solved_tasks (build the invariant).
"""
from __future__ import annotations


def build_contract(solved, unsolved, *, kind: str = "mutation") -> str:
    """The PRESERVE + EXTEND contract text, shared by every generator."""
    solved = list(dict.fromkeys(solved or []))
    unsolved = list(dict.fromkeys(unsolved or []))
    if not solved and not unsolved:
        return ""
    solved_md = "\n".join(f"  - `{t}`" for t in solved) or "  - (none yet)"
    unsolved_md = "\n".join(f"  - `{t}`" for t in unsolved) or "  - (none)"
    extend_verb = (
        "combine the parents so the child additionally"
        if kind == "recombination"
        else "make the harness"
    )
    return (
        "\n## Dynamic-equivalence contract — PRESERVE + EXTEND (read first)\n\n"
        f"This is a preservation-constrained {kind} step, not a free-form "
        "rewrite. The candidate is judged on two things:\n\n"
        "1. **PRESERVE** — the tasks below are already SOLVED by this lineage. "
        "Treat their passing behavior as an INVARIANT: do NOT change, refactor, "
        "or regress the code paths / skills they depend on. A candidate that "
        "breaks ANY of these is rejected before it is scored.\n"
        f"{solved_md}\n\n"
        f"2. **EXTEND** — {extend_verb} solve at least one task it currently "
        "FAILS, WITHOUT disturbing the preserved set. Prefer additive, "
        "narrowly-scoped changes (new skills/branches/fallbacks) over invasive "
        "edits to shared paths the preserved tasks rely on. If a fix would touch "
        "such a path, guard it so existing behavior is unchanged on solved "
        "inputs.\n"
        f"{unsolved_md}\n"
    )


def preserved(candidate_solved, invariant) -> list:
    """Return the invariant tasks the candidate FAILED to preserve (lost).

    Empty list => the candidate preserved the whole (visible) invariant. Callers
    should intersect ``invariant`` with the tasks actually evaluated before
    calling, so an unevaluated invariant task is not counted as lost.
    """
    return sorted(set(invariant or []) - set(candidate_solved or []))


def union_solved(*nodes) -> set:
    """Union of ``.solved_tasks`` across the given nodes (None-safe)."""
    out: set = set()
    for n in nodes:
        if n is None:
            continue
        out |= set(getattr(n, "solved_tasks", None) or [])
    return out


# --- Specialist contract: tiered preserve-and-TRADE (read alongside build_contract) ---
# The flat preserve-extend contract above treats the WHOLE solved union as an
# inviolable invariant: any candidate that loses any solved task is rejected. On a
# frontier that is an all-or-nothing ratchet — every real new capability costs
# *something*, so complementary specialists can never be composed and the merge
# collapses back into the mutation path's monotone behavior.
#
# The specialist contract splits the invariant into two tiers, so the merge can
# make DELIBERATE, BOUNDED trades instead of preserving everything:
#   - CORE       — stably-solved capability (low fragility, broad lineage support).
#                  INVIOLABLE: losing any core task is still a hard reject.
#   - PERIPHERY  — fragile / flip-flop tasks (high variance across evals). TRADEABLE:
#                  a candidate may sacrifice these IFF it is net-positive on periphery
#                  (periphery losses < previously-unsolved tasks gained).
#
# Tier is decided from EVIDENCE (per-task solve history), not a fixed pass-rate
# cutoff — a task at 0.98 mean that still flips is fragile, not core, and only the
# variance/flip history distinguishes it from a genuinely load-bearing 0.98. The
# contract surfaces that evidence so the reasoned verdict draws the line per-case.
# The numeric ``fragility_cutoff`` here is a FALLBACK-ONLY label for the
# verdict-unavailable path (cf. _accept_min_gain), NOT the primary decision.


def tier_tasks(solved, stats_by_task, *, fragility_cutoff: float = 0.35):
    """Split ``solved`` into (core, periphery, evidence) from per-task history.

    ``stats_by_task``: mapping task -> object exposing ``failure_rate`` and
    ``regression_rate`` (e.g. ``tree.TaskOutcomeStats``). A task with no stats is
    treated as CORE (no evidence of fragility → do not risk trading it away).

    ``fragility_cutoff`` is the FALLBACK label only: core = fragility <= cutoff.
    The returned ``evidence`` (per periphery task) is what the reasoned verdict
    reads to judge tradeable-ness; the cutoff is not the primary decision.
    """
    solved = list(dict.fromkeys(solved or []))
    core: list = []
    periphery: list = []
    evidence: dict = {}
    for t in solved:
        s = (stats_by_task or {}).get(t)
        if s is None:
            core.append(t)
            continue
        fr = getattr(s, "failure_rate", 0.0)
        rr = getattr(s, "regression_rate", 0.0)
        # fragility: flips (failures) and regressions relative to how often seen.
        fragility = 0.6 * fr + 0.4 * rr
        if fragility <= fragility_cutoff:
            core.append(t)
        else:
            periphery.append(t)
            evidence[t] = {
                "failure_rate": round(float(fr), 3),
                "regression_rate": round(float(rr), 3),
                "n_evals": int(getattr(s, "total_evals", 0) or 0),
            }
    return core, periphery, evidence


def build_specialist_contract(core, periphery_evidence, targets, *, kind="recombination") -> str:
    """The tiered PRESERVE-CORE + TRADE-PERIPHERY + EXTEND contract text.

    ``core``: inviolable solved tasks. ``periphery_evidence``: {task: {failure_rate,
    regression_rate, n_evals}} — the fragile tasks that MAY be traded, with the
    evidence the verdict uses. ``targets``: previously-unsolved tasks to extend onto.
    """
    core = list(dict.fromkeys(core or []))
    periphery = list((periphery_evidence or {}).keys())
    targets = list(dict.fromkeys(targets or []))
    if not core and not periphery and not targets:
        return ""
    core_md = "\n".join(f"  - `{t}`" for t in core) or "  - (none yet)"
    if periphery:
        peri_md = "\n".join(
            f"  - `{t}` (fails {e.get('failure_rate', 0):.0%} / regresses "
            f"{e.get('regression_rate', 0):.0%} over {e.get('n_evals', 0)} evals)"
            for t, e in periphery_evidence.items()
        )
    else:
        peri_md = "  - (none)"
    targets_md = "\n".join(f"  - `{t}`" for t in targets) or "  - (none)"
    return (
        "\n## Specialist contract — PRESERVE CORE, TRADE PERIPHERY, EXTEND (read first)\n\n"
        f"This is a preservation-constrained {kind} step. Unlike a flat preserve-extend "
        "rewrite, capability is TIERED and periphery may be traded:\n\n"
        "1. **PRESERVE (CORE — inviolable)** — these tasks are solved STABLY across the "
        "lineage. Treat them as a hard invariant: a candidate that loses ANY core task is "
        "rejected before scoring. Do not refactor or regress the paths they depend on.\n"
        f"{core_md}\n\n"
        "2. **TRADE (PERIPHERY — bounded)** — these tasks are solved but FRAGILE (they flip "
        "across evals; the rates below are the evidence). You MAY sacrifice some of them, but "
        "ONLY as a deliberate, NAMED trade that is NET-POSITIVE: the number of periphery tasks "
        "you regress must be strictly LESS than the number of previously-unsolved tasks you "
        "newly solve. Incidental, unjustified periphery loss is rejected.\n"
        f"{peri_md}\n\n"
        "3. **EXTEND** — solve at least one task the lineage currently FAILS, without "
        "disturbing CORE. Prefer additive, narrowly-scoped changes.\n"
        f"{targets_md}\n"
    )


def specialist_preserved(candidate_solved, core, periphery, *, gained) -> tuple[bool, list]:
    """The CHECK for the specialist contract. Returns (ok, lost_core).

    - Lose any CORE task -> (False, [lost core tasks]).
    - Lose CORE=0 and periphery_losses < ``gained`` (net-positive on periphery) -> (True, []).
    - Otherwise (periphery net not positive) -> (False, []).

    ``gained`` is the count of previously-unsolved tasks the candidate newly solves.
    Callers should intersect core/periphery with tasks actually evaluated first, so
    an unevaluated invariant task is not counted as lost.
    """
    solved_set = set(candidate_solved or [])
    lost_core = sorted(set(core or []) - solved_set)
    if lost_core:
        return False, lost_core
    peri_lost = len(set(periphery or []) - solved_set)
    return (peri_lost < int(gained or 0)), []


# --- Routed-code specialization (recombination-via-routing for CODE, not just skills) ---
# A shared CODE edit has unbounded blast radius (cf. a8e25f70's loop.js vocab-broadening
# firing on every task, regressing 14). The fix: make risky code edits ROUTED specialists,
# active ONLY for their target capability-cluster, exactly like conditionally-invoked skills.
# Safety invariants (per external-audit 6.2, NOT assumed — REQUIRED):
#   (1) The non-matching branch MUST be byte-identical to the original code path.
#   (2) The router MUST fail-open to the original path on ANY classification uncertainty.
#   (3) Route on the EXISTING capability-cluster taxonomy (generalization._task_cluster),
#       do not invent a new per-edit classifier.
# The gate then verifies the routing predicate EMPIRICALLY (does the guard fire outside the
# claimed target cluster on the canary/guard set?) — the code analogue of the whenToUse check.
def routed_code_contract(target_cluster: str | None = None) -> str:
    tc = target_cluster or "<the claimed target capability-cluster>"
    return (
        "\n## Routed-code contract — SPECIALIZE, do not rewrite shared flow (read if editing code)\n\n"
        "If your fix requires editing shared engine/control-flow code (not a pure additive skill), "
        "it MUST be a ROUTED specialist, not an unconditional edit:\n"
        f"1. GUARD the new behavior on the task's capability-cluster == '{tc}' "
        "(use the existing cluster taxonomy; do NOT invent a new classifier).\n"
        "2. The ELSE branch MUST be BYTE-IDENTICAL to the original code — unrelated clusters run "
        "the unchanged path, so their behavior is provably preserved.\n"
        "3. FAIL-OPEN: on any uncertainty about the task's cluster, take the ORIGINAL path.\n"
        "An UNCONDITIONAL shared-code edit (no cluster guard) is rejected: it has unbounded blast "
        "radius and is the exact mechanism that regressed 14 tasks while fixing 7. If a global "
        "improvement genuinely helps ALL clusters with zero regressions, say so explicitly and it "
        "may stay unconditional — otherwise, route it.\n"
    )
