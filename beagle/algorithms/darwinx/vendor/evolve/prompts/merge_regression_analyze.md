# Analyze merge regression

You are in read-only plan mode for a self-evolve merge repair.

- Worktree: `{{ wt_dir }}`
- Pipeline: `{{ pipeline_id }}`
- Repair iteration: `{{ iteration }} / {{ max_iters }}`
- Primary parent: `{{ primary_parent }}` score={{ primary_score }}
- Secondary parent: `{{ secondary_parent }}` score={{ secondary_score }}
- Merged child score: {{ child_score }}
- Child eval logs: `{{ job_log_path }}`
- Primary parent eval logs: `{{ primary_job_log_path }}`
- Secondary parent eval logs: `{{ secondary_job_log_path }}`

## Parent / Child Pass-Fail Delta

```json
{{ parent_child_delta_json }}
```

## Merge Contract

The child must preserve every parent win before it can be promoted.

```json
{{ merge_contract_json }}
```

## Repair Context

This includes the latest validation coverage, exact lost wins, prior repair
notes, and the best rejected merge candidate if one exists.

```json
{{ merge_repair_context_json }}
```

Analyze why the merged child does not outperform both parents. Decide whether
the regression/no-improvement is likely solvable by a general `monet_code`
change on top of the merged child.

For every proposed repair, state which primary-only, secondary-only, or
both-parent win it preserves or restores. A repair that fixes one child failure
by losing another parent win should be treated as not useful unless it produces
a strictly better child overall.

If it is not solvable, include the exact token `UNSOLVABLE`.
If it is solvable, provide a concrete repair plan. Do not propose task-specific
shortcuts or benchmark identity checks.
