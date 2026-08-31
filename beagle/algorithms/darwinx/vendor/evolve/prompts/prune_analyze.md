# Simplify the agent by REMOVING accumulated modifications

You are improving `monet_code` in `{{ wt_dir }}/monet_code/`.

**This is a PRUNE node. Your job is SUBTRACTION, not addition.** Every other node
in this campaign adds something. You do the opposite: you find modifications that
earlier nodes accumulated and you REMOVE the ones that are dead, redundant,
never-triggering, or actively harmful — while keeping everything that is doing
real work.

Context:
- Campaign dir: `{{ campaign_dir }}`
- Pipeline: `{{ pipeline_id }}`
- Iteration: `{{ iteration }}` of `{{ max_iters }}`
- Parent commit: `{{ parent_commit }}`
- Branch: `{{ monet_branch }}`
- Lineage under review: `{{ node_ids_to_investigate | join(", ") }}`
- Tasks currently claimed/measured: `{{ claimed_tasks | join(", ") }}`
- Eval basis: `{{ eval_kind }}`
- Eval/job directory to start from: `{{ input_eval_dir }}`
{{ prune_dossier_block }}
{{ collective_knowledge_block }}
{{ recent_lessons_block }}

{% if trace_qc_summary %}
Automated trajectory QC of the parent's current failures. Read this for evidence
that an accumulated modification is *causing* a failure — a skill that fires on
the wrong cue, a guard that swallows an error, a branch that sends the agent down
a worse path. Such a modification is the best possible prune target.

{{ trace_qc_summary }}

{% endif %}
You are in read-only plan mode. Produce a plan that REMOVES exactly one coherent
accumulated modification.

## Why subtraction matters

An evolved agent accumulates modifications monotonically: every accepted node
added a skill or a guarded code branch, and nothing was ever removed. That decay
is real and measurable:

- **Dead weight** — a cue-gated skill whose trigger never fires on real tasks
  costs prompt budget and attention on every single task while helping none.
- **Redundancy** — several nodes independently added overlapping procedures for
  the same situation. They now compete, and the agent picks arbitrarily.
- **Active harm** — a modification that helped the one task it was written for
  but quietly costs tasks elsewhere. It was accepted because the screening slice
  did not cover the tasks it hurts.
- **Blocked evolution** — an over-narrow guard added early can prevent a later,
  better general fix from ever taking effect.

Removing such a modification is a genuine improvement even when the score does
not move: the agent gets simpler, cheaper, and easier to improve next.

## What counts as a good prune target

Rank candidates by the evidence you can actually cite:

1. **Provably harmful** — trajectory evidence that the modification fires and
   leads to a worse outcome, or that it suppresses an error the agent needed to
   see. Removing it should *raise* the score. Best target by far.
2. **Provably dead** — the cue/guard cannot fire, or never fired across the
   trajectories available to you. Removing it is behavior-preserving by
   construction.
3. **Redundant** — two or more modifications cover the same situation; consolidate
   by deleting the weaker/narrower one and leaving the general one intact.
4. **Over-narrow** — a task-shaped special case that a later general mechanism now
   subsumes.

## What you MUST NOT remove

These are hard constraints. A plan that violates them will be rejected.

- **Anything load-bearing.** If a currently-passing task depends on it, leave it
  alone. The tasks listed as solved in the dossier are an invariant: your change
  must not regress them. When in doubt about whether something is load-bearing,
  pick a different target — an unnecessary regression is much worse than a
  missed prune.
- **Tests, verifiers, or task-provided checks.** Never delete or weaken a test
  file, an assertion, a verification step, or an evidence/audit check in order to
  make something pass. That is eval tampering and it poisons the node.
- **Stock product functionality.** The base agent's own features (its shipped
  skills, its provider/gateway plumbing, its CLI surface) are not evolution
  artifacts. Only remove things this campaign's lineage ADDED, as listed in the
  dossier above.
- **More than one coherent thing.** Remove ONE modification so the result is
  attributable. Do not bundle a removal with an addition or a refactor — a prune
  node's diff should be dominated by deletions.

## How you will be judged

Unlike every other node, you are NOT required to make the score go up. A prune
is a success when **behavior is preserved and the agent got smaller**, and a
strong success when **removing the modification also fixed something**. State
explicitly in your plan which of these you expect, and why.

So your plan must include, in whatever structure is natural:

- **The target**: exactly what you will delete, by file and symbol/entry, and
  which lineage commit or node introduced it.
- **The evidence**: why you believe it is harmful, dead, redundant, or
  over-narrow — citing the dossier, the trajectories, or the code itself. Say
  plainly if your evidence is weak.
- **The load-bearing argument**: why removing it cannot regress the currently
  solved tasks. Name the tasks that traverse the nearest code path and explain
  why they are unaffected.
- **The expected outcome**: score-preserving simplification, or a score gain.

Use plan mode to write the plan. Include enough detail for the next
implementation step to apply the removal safely and completely — a partial
deletion that leaves a dangling reference is a broken agent, so specify every
call site, registry entry, import, and test that must be updated.
