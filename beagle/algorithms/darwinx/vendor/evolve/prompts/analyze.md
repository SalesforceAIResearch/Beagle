# Analyze failures and propose a general fix

You are improving `monet_code` in `{{ wt_dir }}/monet_code/`.

Context:
- Campaign dir: `{{ campaign_dir }}`
- Pipeline: `{{ pipeline_id }}`
- Iteration: `{{ iteration }}` of `{{ max_iters }}`
- Parent commit: `{{ parent_commit }}`
- Branch: `{{ monet_branch }}`
- Nodes to investigate: `{{ node_ids_to_investigate | join(", ") }}`
- Claimed tasks: `{{ claimed_tasks | join(", ") }}`
- Eval basis: `{{ eval_kind }}`
- Eval/job directory to start from: `{{ input_eval_dir }}`
{{ preserve_extend_block }}
{{ task_contract_block }}
{{ collective_knowledge_block }}
{{ recent_lessons_block }}
{{ contrastive_block }}

{% if shared_experience_summary %}
Shared experience from this campaign:

```text
{{ shared_experience_summary }}
```

{% endif %}
{% if trace_qc_summary %}
Automated trajectory QC — structured, evidence-cited findings to start from.
These are candidate symptoms, not verdicts (and an empty result is not proof of
a clean run); confirm each against the trajectory it cites before acting. The
block lists, per task, the normalized `*.messages.jsonl` to open for the cited
`#message_index` (the raw SSE stream is the `*.trajectory.jsonl` beside it).

{{ trace_qc_summary }}

{% endif %}
You are in read-only plan mode. Review the trajectories and logs for each
claimed task carefully, identify the general `monet_code` issue, and propose a
plan to resolve it. The fix or improvement must generalize across tasks and
must not overfit to benchmark task names, expected outputs, verifier strings,
or specific trial ids.{% if shared_experience_summary %} Shared experience may be inaccurate; use it to avoid
duplicated work, but inspect original logs when needed.{% endif %}

Plan focused offline validation only. Avoid broad provider/API integration
suites unless the plan truly requires them.

## High-leverage scaffold directions (frontier gap)

On this benchmark the same base model scores ~5 points higher under a stronger
agent scaffold than a weaker one, so the biggest wins come from general
agent-behaviour improvements, not task-specific patches. When the failure
evidence points that way, prefer fixes in these directions (all must
generalize, never hardcode task specifics):

- Verification discipline: before declaring a task done, exercise the real /
  task-provided tests or the true public entrypoint of the artifact, not a
  self-authored smoke check. Treat "looks done" as unverified.
- Long-horizon robustness: for multi-step, compute-heavy, or build/training
  tasks, decompose into checkpointed subgoals, verify intermediate artifacts,
  and recover from partial failure instead of restarting or giving up.
- Persistent working memory: on long tasks, have the agent keep durable notes
  on disk (plan, progress, what was tried/ruled out) and re-read them rather
  than relying on a context window that gets summarized away. Frontier models
  gain far more from file-based memory on long-horizon work than from raw
  context length.
- Timeout-robust execution: prefer non-blocking command execution that polls
  for completion and returns partial state on timeout, over a model that blocks
  waiting on a single long command — blocking execution inflates spurious
  timeouts and adds score noise.
- Tool-use reliability: prefer idempotent commands, re-read on-disk/system
  state after mutating actions, and handle non-zero exit codes explicitly
  rather than assuming success.
- Budget awareness: on long tasks, land the highest-value subgoal early and
  avoid spending the whole turn/time budget on exploration.

### Highest-leverage direction: close the best-of-k vs avg gap (THE primary lever)

Measured evidence on this benchmark: the agent's **best-of-2 score is far above
its avg@k score**, and the dominant failure (~73% of fails) is *genuine
wrong-answers* — monet finishes and declares success but the answer is wrong
(right file, wrong semantics; plausible-but-incorrect artifact) — NOT timeouts
or infra. This means monet very often *can* produce a correct solution but its
**modal rollout is wrong while a minority rollout is right**. The single biggest
win is therefore to make monet's *own* rollout reliably land on the
correct-rollout it is already capable of — i.e. internalize the selection that
best-of-k does externally, so it works in a single graded attempt:

- Solve-time verification contract: before declaring done, monet should derive
  an explicit, checkable acceptance contract from the task (what observable
  behaviour/output must hold), then *prove* its candidate solution satisfies it
  using the real task entrypoint/tests — never a self-authored smoke check that
  trivially passes. "Looks done" must count as unverified.
- Generate→verify→select / retry: when the contract is not satisfied (or the
  result is uncertain), monet should revise or produce an alternative candidate
  and re-verify, keeping the candidate that provably satisfies the contract,
  rather than committing the first plausible attempt. This is what converts
  latent best-of-k capability into a reliable single submission.
- Make verification cheap and general so it scales to every task (no hardcoding
  task specifics): infer the contract from the problem statement + repo/task
  conventions, reuse existing tests/entrypoints, and prefer the least-disruptive
  change consistent with those conventions.

Prefer fixes in this direction when the evidence shows wrong-but-confident
completions; it attacks the largest measured headroom directly.

### Hard constraint: make the change ADDITIVE, not a rewrite of shared code

Measured failure mode (prior campaign, 13 pipelines): every rejected candidate
fixed its claimed tasks but **regressed 13-24 other tasks**, because it MODIFIED
monet's shared core — the agent loop, tool/command dispatch, shell-execution
handling — which the many currently-passing build/compile/git/systems tasks all
depend on. Improvement and regression were coupled through that shared path.

Therefore your fix MUST be additive and locality-bounded:

- **Add** new, conditionally-guarded code paths / helpers for the target
  behaviour; do **not** rewrite or delete existing functions on the shared
  execution/tool-dispatch path that passing tasks traverse.
- Aim for a near-pure-addition diff (think `+N / -0..1`, like a new branch +
  a new helper), not a refactor of core logic. A large number of deleted /
  rewritten existing lines is the signal of the failure mode above and will be
  rejected before evaluation.
- If the only way you can see to fix the task is to change shared core
  behaviour, prefer a **narrow guard** (detect the situation that needs the new
  behaviour and branch into your new path) so the default path that passing
  tasks rely on is byte-for-byte unchanged.

This is the single highest-leverage constraint: a smaller, additive, correct
change that preserves the base beats a larger one that regresses it.

### Two surfaces you can improve: CORE vs SKILLS — edit exactly ONE, targeted

monet has two independent surfaces. **Do not mix them in one candidate** — they must
be separately attributable so each is judged on its own. Choose the surface by where
the fault actually is, and **prefer the change that generalizes to unseen tasks**: you
are judged on held-out tasks you cannot see, so a fix that only helps the exact task
you inspected will not score.

- **CORE** (`src/` — the agent loop, planning / turn control, tool & command dispatch,
  verification, shell & error/recovery handling). Because the core is traversed by
  every task, a genuine core improvement is the **highest-value, best-generalizing**
  change — it lifts many tasks at once. This is a **first-class target**: if the fault
  is in how the agent *reasons, plans, dispatches tools, recovers from failures, or
  verifies its work*, improve the core. Keep it disciplined — prefer additive or
  guarded changes and preserve behaviour that passing tasks rely on — but **do NOT
  avoid the core**. A real loop / tool-dispatch / verification / recovery improvement
  is worth more than a task-specific skill.
- **SKILLS** — a reusable procedure added to the `BUNDLED_SKILLS` array in
  `src/core/bundled-skills.js` (the only skill path that PERSISTS; `.monet/skills/` is
  runtime-only and will NOT ship). A skill is additive and low regression risk, **but
  beware its failure mode: a cue-gated skill only activates on its trigger, so it tends
  to help ONLY the task it was written for and does NOT generalize to held-out tasks.**
  Use a skill only when the gap is genuinely recurring task-type know-how, and make it
  GENERAL (no task-name literals, no expected-output strings, no narrow one-off cues).

Use the fault analysis (trace QC) to decide which surface the fault is on:
- the *engine itself* mishandles reasoning / planning / tool-dispatch / verification /
  recovery that many tasks traverse → improve the **CORE** (preferred when it
  generalizes — this is the intended path for real self-improvement); or
- a genuinely recurring task-type procedure is missing → add a **general SKILL**.

State which surface you are editing and why, then make a targeted edit to **only that
surface** — and prefer the change that lifts unseen tasks, not just the one in front of you.

Use plan mode to write a clear implementation plan in whatever structure is
most natural. Include enough detail for the next implementation step to apply
the fix safely.
