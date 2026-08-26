# Summarize learnings from this self-evolve branch

You just shepherded a self-evolve pipeline to a successful improvement
on the **{{ campaign }}** campaign. Score went from `{{ parent_score }}`
to `{{ child_score }}` (delta: `{{ delta }}`) over `{{ n_iterations }}`
iterations. The branch `{{ monet_branch }}` was just opened as a PR
({{ pr_url or "PR pending" }}).

## Inputs

- Campaign: `{{ campaign }}`
- Pipeline: `{{ pipeline_id }}`
- Subset: `{{ subset }}`
- Targeted tasks: `{{ claimed_tasks | join(", ") }}`
- Resolved tasks (now passing that weren't before): `{{ resolved_tasks | join(", ") or "(none reported)" }}`
- Effort log (commit-by-commit, with mini-eval results per iteration):
  `{{ effort_md_path }}`

## What to write

Produce a short **learnings.md** (300–600 words) for this branch. The
goal is to capture insights that the next pipeline picking this commit
as a parent (or a human reviewer of the PR) can act on.

Structure:

```markdown
# Learnings from `{{ monet_branch }}`

## TL;DR

<2 sentences: the bug class fixed and how broadly it generalizes.>

## What worked

- <pattern that landed cleanly, with the iteration # it landed in>
- ...

## What didn't work (and why)

- <approach tried that was reverted, plus the failure mode>
- ...

## Generalization assessment

<one paragraph: which other failures in the broader benchmark this
change should help, and which it definitely won't.>

## Recommended next moves

- <next-step #1 for someone branching from this commit>
- ...
```

Reply with **only the markdown** for `learnings.md` — no surrounding
prose, no explanation. The orchestrator saves your reply verbatim.
