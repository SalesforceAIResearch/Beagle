# Consolidate the harness: same capability, less machinery

You are improving `monet_code` in `{{ wt_dir }}/monet_code/`.

**This is a CONSOLIDATE node. Your job is a REWRITE for simplicity, not an
addition and not a deletion.** Other nodes add a capability. Prune nodes delete
dead weight. You do the third thing, the one that actually makes a harness good:
you find places where the agent has accumulated several overlapping mechanisms
and you replace them with one general mechanism that covers all of them — and
you are explicitly allowed to rewrite the ORIGINAL pre-evolve code to do it.

Nothing here is off-limits except the contracts listed under "What you must not
break". The base agent's own design is not sacred: if the campaign has bolted
three special cases onto a structure that was never shaped for them, the right
move is usually to reshape the structure, not to add a fourth special case.

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
Automated trajectory QC of the parent's current failures. Read it for evidence
that accumulated machinery is now *in the way* — instructions that contradict
each other, a guard that swallows the error the agent needed to see, two
overlapping rules where the agent visibly cannot tell which applies.

{{ trace_qc_summary }}
{% endif %}

## The two surfaces

- **SKILLS / accumulated procedures** (the `BUNDLED_SKILLS` entries). Accumulation
  shows up here first: several narrow, cue-gated skills that overlap. Consolidating
  means replacing them with one general procedure whose cue covers all their cases.
- **CORE** (`src/` — the agent loop, tool/command dispatch, verification, recovery).
  Consolidating here means collapsing branches: where several guarded paths handle
  variants of one situation, replace them with the general case. You MAY rewrite
  pre-evolve core code to do this; that is what separates this node from a prune.

## What counts as success

Two things must BOTH hold, and you are measured on both:

1. **Capability is preserved or improved.** Every task the parent solves must
   still be solved. This is not negotiable and is checked by evaluation.
2. **The harness is measurably simpler.** Fewer total lines, or materially fewer
   branch points at similar length. The branch reduction is the stronger signal —
   collapsing three special cases into one general path is the intended move even
   if the line count barely changes.

A rewrite that is bigger and no better is the exact failure this node exists to
avoid. A rename, a reflow, or moving code between files is not a consolidation.

## What you MUST NOT break

- **Tests, assertions, verifiers, evidence checks.** Never delete or weaken one.
  Simplifying by deleting the thing that measures you is the single worst outcome
  here and will be treated as tampering, not as a consolidation.
- **Harness contracts.** The submission mechanism (how the agent signals it is
  done and emits its patch), the step and cost limits, and the shape of the
  observation returned to the model. You may re-express these more cleanly; you
  may not remove them or change what the outside world sees.
- **Capability you cannot prove is redundant.** The danger of a rewrite is losing
  behaviour that only matters on tasks outside the measured slice. If your
  argument for equivalence is "this looks unused", that is not sufficient — say
  so honestly and pick a target you can actually argue from the trajectories.

## Your plan

You are in read-only plan mode. Produce a plan for ONE coherent consolidation.
Include, in whatever structure is natural:

- **The target**: the specific mechanisms being folded together, by file and
  symbol/section, and what single general mechanism replaces them.
- **The equivalence argument**: for each behaviour currently produced by the old
  mechanisms, where it comes from afterwards. This is the load-bearing part of the
  plan — it is what distinguishes a consolidation from a capability deletion.
- **The expected complexity change**: roughly how many lines and branches this
  removes, and which of the two is the real win.
- **What you deliberately are NOT touching**, and why.
