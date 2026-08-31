# Analyze failures and propose a general fix

You are improving `agents/mini_swe_agent/vendor` in `{{ wt_dir }}/agents/mini_swe_agent/vendor/`.

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
claimed task carefully, identify the general `agents/mini_swe_agent/vendor` issue, and propose a
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
wrong-answers* — mini-swe-agent finishes and declares success but the answer is wrong
(right file, wrong semantics; plausible-but-incorrect artifact) — NOT timeouts
or infra. This means mini-swe-agent very often *can* produce a correct solution but its
**modal rollout is wrong while a minority rollout is right**. The single biggest
win is therefore to make mini-swe-agent's *own* rollout reliably land on the
correct-rollout it is already capable of — i.e. internalize the selection that
best-of-k does externally, so it works in a single graded attempt:

- Solve-time verification contract: before declaring done, mini-swe-agent should derive
  an explicit, checkable acceptance contract from the task (what observable
  behaviour/output must hold), then *prove* its candidate solution satisfies it
  using the real task entrypoint/tests — never a self-authored smoke check that
  trivially passes. "Looks done" must count as unverified.
- Generate→verify→select / retry: when the contract is not satisfied (or the
  result is uncertain), mini-swe-agent should revise or produce an alternative candidate
  and re-verify, keeping the candidate that provably satisfies the contract,
  rather than committing the first plausible attempt. This is what converts
  latent best-of-k capability into a reliable single submission.
- Make verification cheap and general so it scales to every task (no hardcoding
  task specifics): infer the contract from the problem statement + repo/task
  conventions, reuse existing tests/entrypoints, and prefer the least-disruptive
  change consistent with those conventions.

Prefer fixes in this direction when the evidence shows wrong-but-confident
completions; it attacks the largest measured headroom directly.

### Hard constraint: change the smallest thing that could work

mini-swe-agent is ~370 lines in total: a 190-line agent loop plus a 183-line
prompt config. Nothing here is peripheral — every task traverses the same loop —
so an "additive-only" rule would forbid essentially every real improvement. The
discipline is therefore *smallness and reversibility*, not addition:

- Make the smallest edit that could plausibly move the measured failure mode.
- Preserve the contracts the harness depends on: the submission sentinel, the
  step and cost limits, and the shape of the observation dict returned to the
  model. Breaking one of these does not lower the score, it voids the run.
- Never add task-name literals, expected-output strings, or cues copied from the
  logs. You are scored on held-out tasks you cannot see.
- Changes that regress currently-passing tasks are caught by the canary gate and
  reverted, so prefer changes whose downside is bounded and legible.

### Two surfaces you can improve: PROMPT vs LOOP — edit exactly ONE, targeted

- **PROMPT** (`src/minisweagent/config/mini.yaml`) — the system and
  instance templates: the workflow the model is told to follow, its boundaries, how
  it is asked to confirm a fix before submitting. For a minimal agent this is often
  the highest-leverage surface, because the scaffold's behaviour largely *is* its
  instructions. Edits generalize when they encode a better *general* procedure
  (reproduce, fix, re-run, check edge cases) and fail when they encode one task's
  answer.
- **LOOP** (`src/minisweagent/agents/default.py`) — step control, message and
  observation construction, error and format-error handling, submission detection,
  limit enforcement. Edit this when the fault is structural: the agent gives up
  early, mishandles a failed command, wastes steps, or submits unverified. This is
  a first-class target — do not avoid it — but keep the edit tight.

Use the fault analysis to decide which surface the fault is on:
- the model is told the wrong procedure, or is never told to verify → **PROMPT**;
- the loop structurally prevents recovery, verification, or completion → **LOOP**.

State which surface you are editing and why, then edit only that surface.

Use plan mode to write a clear implementation plan in whatever structure is
most natural. Include enough detail for the next implementation step to apply
the fix safely.
