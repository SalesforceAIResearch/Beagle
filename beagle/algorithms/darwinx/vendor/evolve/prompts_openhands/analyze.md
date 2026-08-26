# Analyze failures and propose a general fix

You are improving `agents/openhands/vendor` in `{{ wt_dir }}/agents/openhands/vendor/`.

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
claimed task carefully, identify the general `agents/openhands/vendor` issue, and propose a
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
wrong-answers* — openhands finishes and declares success but the answer is wrong
(right file, wrong semantics; plausible-but-incorrect artifact) — NOT timeouts
or infra. This means openhands very often *can* produce a correct solution but its
**modal rollout is wrong while a minority rollout is right**. The single biggest
win is therefore to make openhands's *own* rollout reliably land on the
correct-rollout it is already capable of — i.e. internalize the selection that
best-of-k does externally, so it works in a single graded attempt:

- Solve-time verification contract: before declaring done, openhands should derive
  an explicit, checkable acceptance contract from the task (what observable
  behaviour/output must hold), then *prove* its candidate solution satisfies it
  using the real task entrypoint/tests — never a self-authored smoke check that
  trivially passes. "Looks done" must count as unverified.
- Generate→verify→select / retry: when the contract is not satisfied (or the
  result is uncertain), openhands should revise or produce an alternative candidate
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

openhands is a large SDK, not a 370-line script, so "small" here is a stronger
discipline than it was for a minimal agent: almost anything you touch is on the
path of every task, and the blast radius of a careless edit is the whole panel.

- Make the smallest edit that could plausibly move the measured failure mode.
- Preserve the contracts the harness depends on: the tool-call protocol, the
  finish/completion signal, the conversation and event shapes, and the
  serialisation boundaries other packages import. Breaking one of these does not
  lower the score, it voids the run — every task errors and the node is wasted.
- Never add task-name literals, expected-output strings, or cues copied from the
  logs. You are scored on held-out tasks you cannot see.
- Changes that regress currently-passing tasks are caught by the canary gate and
  reverted, so prefer changes whose downside is bounded and legible.

### Three surfaces you can improve — edit exactly ONE, targeted

Only two packages are installed into the trial container, `openhands-sdk/` and
`openhands-tools/`. **Editing `openhands-agent-server/` changes nothing about
the score**: it is not installed. Confirm any file you plan to edit is under one
of the two installed packages before planning the edit.

- **PROMPT** (`openhands-sdk/openhands/sdk/context/prompts/sections/static.py`,
  composed in `.../prompts/presets.py`) — the system prompt is built from named
  sections rather than one template file, so a prompt edit means changing the
  text of a section, or which sections the preset composes. Often the
  highest-leverage surface: much of a scaffold's behaviour is its instructions.
  Edits generalize when they encode a better *general* procedure (reproduce,
  fix, re-run, check edge cases) and fail when they encode one task's answer.
- **LOOP** (`openhands-sdk/openhands/sdk/agent/agent.py`, and `agent/base.py`)
  — step control, message and observation construction, tool dispatch, error and
  malformed-call handling, finish detection, limit enforcement. Edit this when
  the fault is structural: the agent gives up early, mishandles a failed
  command, wastes steps, or finishes unverified. A first-class target — do not
  avoid it — but keep the edit tight, and note it is ~1400 lines, so read the
  region you are changing before changing it.
- **TOOLS** (`openhands-tools/openhands/tools/`: `terminal`, `file_editor`,
  `task_tracker`, `grep`, `glob`, ... selected by `preset/default.py`) — what
  the agent can actually do and what it observes back. Edit this when the fault
  is in the *interface*: a tool returns output the model misreads, truncates
  something it needed, fails silently on an error the agent should have seen, or
  the preset withholds a capability the failures show it needs.

Use the fault analysis to decide which surface the fault is on:
- the model is told the wrong procedure, or is never told to verify → **PROMPT**;
- the loop structurally prevents recovery, verification, or completion → **LOOP**;
- the agent acts sensibly but gets back a misleading or unusable observation, or
  lacks a capability entirely → **TOOLS**.

State which surface you are editing and why, then edit only that surface.

Use plan mode to write a clear implementation plan in whatever structure is
most natural. Include enough detail for the next implementation step to apply
the fix safely.
