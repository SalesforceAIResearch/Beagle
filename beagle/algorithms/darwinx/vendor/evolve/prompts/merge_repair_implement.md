# Implement merge regression repair

You are working in `{{ wt_dir }}/monet_code/` on a merged self-evolve child.

- Pipeline: `{{ pipeline_id }}`
- Repair iteration: `{{ iteration }} / {{ max_iters }}`

Use the analysis and plan below to implement a general repair. Edit only
`monet_code/`. Do not edit the eval harness, and do not introduce
task-specific hacks.

Run focused, offline checks only. Do not run broad provider/API integration
suites such as full `npm test` or `tests/output-format.test.js`; those can
fail from external credentials/model routing and are not merge-repair evidence.

## Merge Contract

```json
{{ merge_contract_json }}
```

## Repair Context

```json
{{ merge_repair_context_json }}
```

## Analysis

{{ analysis }}

## Plan

{{ plan_text }}

When done, leave the working tree ready to commit.
