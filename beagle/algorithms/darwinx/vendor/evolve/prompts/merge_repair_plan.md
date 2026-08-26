# Plan merge regression repair

You are in read-only plan mode for a self-evolve merge repair.

- Worktree: `{{ wt_dir }}`
- Pipeline: `{{ pipeline_id }}`
- Repair iteration: `{{ iteration }} / {{ max_iters }}`

Create a concrete, general repair plan based on the regression analysis below.
The plan must target `monet_code/` only and must not use task-specific hacks.

## Merge Contract

```json
{{ merge_contract_json }}
```

## Repair Context

```json
{{ merge_repair_context_json }}
```

## Regression Analysis

{{ analysis }}

For validation, plan focused offline checks only. Avoid full `npm test`,
`tests/output-format.test.js`, and other provider/API integration suites
because they can fail from external credentials/model routing unrelated to the
merge repair.

Return a concise markdown plan with the files/systems to inspect, the intended
behavioral change, and the validation you expect the implementation to run.
