# Self-review merge regression repair

You are reviewing an uncommitted repair in `{{ wt_dir }}/monet_code/`.

- Pipeline: `{{ pipeline_id }}`
- Repair iteration: `{{ iteration }} / {{ max_iters }}`

Review the diff against the plan. Check for obvious bugs, overfitting,
task-specific conditionals, missing tests, and whether the change plausibly
addresses the regression without harming either parent behavior or any child
new wins that should be preserved.

Use focused, offline validation only. Do not run full `npm test`,
`tests/output-format.test.js`, or other provider/API integration suites; those
depend on external credentials/model routing and should not decide whether a
merge repair is valid.

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

## Repair Plan

{{ plan_text }}

Return a concise review and end with exactly one marker:

```text
<<<REPAIR_REVIEW>>> APPROVE
```

or

```text
<<<REPAIR_REVIEW>>> BLOCK
```

Use `BLOCK` if the repair clearly violates the merge contract, overfits to a
benchmark identity, or risks losing a parent win.
