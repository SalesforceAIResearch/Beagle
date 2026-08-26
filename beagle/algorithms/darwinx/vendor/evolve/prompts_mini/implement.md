# Implement the plan

Implement the plan at `{{ plan_path }}` in `{{ wt_dir }}/agents/mini_swe_agent/vendor/`.

Context: pipeline `{{ pipeline_id }}`, iteration `{{ iteration }} / {{ max_iters }}`,
branch `{{ monet_branch }}`, parent commit `{{ parent_commit }}`.

{{ preserve_extend_block }}

Only edit `{{ wt_dir }}/agents/mini_swe_agent/vendor/`; do not edit the eval harness, scripts, configs,
task files, or benchmark verifier behavior. Keep the change general: do not add task
name literals, trial ids, `/app/<task>` gates, expected-output strings, or verifier
text copied from the logs.

Run focused offline checks that match the plan. Do not commit, do not push, and do
not spend time on broad provider/API integration suites in this phase.

When done, reply with a short summary of what changed and where.
