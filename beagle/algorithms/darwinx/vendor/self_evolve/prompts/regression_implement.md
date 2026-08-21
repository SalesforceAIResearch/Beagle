# Implement the regression resolver plan

Implement the plan at `{{ plan_path }}` in `{{ wt_dir }}/monet_code/`.

Context: pipeline `{{ pipeline_id }}`, iteration `{{ iteration }} / {{ max_iters }}`,
target node `{{ target_node_id }}`, branch `{{ monet_branch }}`, parent commit
`{{ parent_commit }}`. Regressions to fix: `{{ claimed_tasks | join(", ") }}`.
Preserve target improvements `{{ target_improved_tasks | join(", ") }}` and
parent-solved tasks `{{ target_solved_tasks | join(", ") }}`.

Only edit `{{ wt_dir }}/monet_code/`. Do not edit the eval harness, scripts,
configs, Harbor wrappers, benchmark timeouts, verifier behavior, or task files.
Keep the change general: no task-name literals, trial ids, `/app/<task>` gates,
expected-output strings, or copied verifier text.

Do not commit or push. Run focused offline checks only; the review step handles
commit and broader validation.
