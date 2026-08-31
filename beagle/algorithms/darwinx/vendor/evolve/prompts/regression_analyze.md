# Analyze regressions and preserve improvements

You are in read-only plan mode for a regression resolver in `{{ wt_dir }}/monet_code/`.

Context:
- Campaign dir: `{{ campaign_dir }}`
- Pipeline: `{{ pipeline_id }}`
- Iteration: `{{ iteration }}` of `{{ max_iters }}`
- Target node: `{{ target_node_id }}`
- Target score: `{{ target_score }}`
- Parent/target commit: `{{ parent_commit }}`
- Resolver branch: `{{ monet_branch }}`
- Regressed tasks to fix: `{{ claimed_tasks | join(", ") }}`
- Improvements to preserve: `{{ target_improved_tasks | join(", ") }}`
- Parent-solved tasks to protect: `{{ target_solved_tasks | join(", ") }}`
- Eval/job directory to start from: `{{ input_eval_dir }}`

{% if shared_experience_summary %}
Shared experience from this campaign:

```text
{{ shared_experience_summary }}
```

{% endif %}
Review the trajectories and logs for the regressed tasks, identify the general
`monet_code` failure mode, and propose a plan that fixes as many regressions as
possible without breaking parent-solved behavior. The plan must edit only
`monet_code/`, must not overfit to task names or verifier strings, and should
{% if shared_experience_summary %}use shared experience only as advisory context because summaries and evals can
be noisy.{% else %}generalize across similar failure modes without relying on task-specific details.{% endif %}

Use plan mode to write a clear implementation plan in whatever structure is
most natural. Include enough detail for the next implementation step to apply
the fix safely, including how the change protects parent-solved behavior.
