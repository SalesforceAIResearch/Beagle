# Resolve a self-evolve merge

You are working in an isolated `monet_code_eval` worktree:

- Worktree: `{{ wt_dir }}`
- `monet_code` branch: `{{ monet_branch }}`
- Pipeline: `{{ pipeline_id }}`
- Primary parent node: `{{ primary_parent }}`
- Secondary parent node: `{{ secondary_parent }}`
- Secondary commit: `{{ secondary_commit }}`

The orchestrator attempted to merge the secondary parent into the child branch.
Git reported:

```text
{{ git_stdout }}
{{ git_stderr }}
```

## Merge Contract

The merged child is only useful if it preserves the union of parent wins and
then improves beyond both parents. Use this contract while resolving semantic
conflicts; do not treat this as task-specific shortcut permission.

```json
{{ merge_contract_json }}
```

Resolve the merge in `{{ wt_dir }}/monet_code/`. Do not edit the eval harness.
Preserve the useful behavior from both parents and avoid task-specific hacks.
Run focused, offline checks only. Do not run broad provider/API integration
suites such as full `npm test` or `tests/output-format.test.js`; those can
fail from external credentials/model routing and are not merge-resolution
evidence.
When done, leave the working tree ready to commit.
