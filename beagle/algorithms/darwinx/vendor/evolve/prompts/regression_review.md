# Review regression resolver changes

Review the regression resolver diff in `{{ wt_dir }}/monet_code/`, run focused
checks, fix bugs you introduced, and commit only if the change is general and
does not plausibly break parent-solved behavior.

Context:
- Pipeline: `{{ pipeline_id }}`
- Iteration: `{{ iteration }} / {{ max_iters }}`
- Branch: `{{ monet_branch }}`
- Base commit: `{{ parent_commit }}`
- Regressed tasks being fixed: `{{ claimed_tasks | join(", ") }}`
- Target improvements to preserve: `{{ target_improved_tasks | join(", ") }}`
- Parent-solved tasks to protect: `{{ target_solved_tasks | join(", ") }}`

Checklist:
1. Inspect `git -C {{ wt_dir }}/monet_code diff {{ parent_commit }}..HEAD`.
2. Confirm the diff edits only `monet_code/`.
3. Reject task-specific hacks: no task-name literals, trial ids, copied verifier
   strings, `/app/<task>` gates, or `if task == ...` narrowing conditionals.
4. Run focused offline checks for the changed behavior and preservation risk.
5. Commit only after checks pass.

In the final reply, include `## Generalization Audit`, `## Validation`, and:

```markdown
## Shared Experience Updates
- task: `<task>`; kind: improved|regressed; commit: `<sha>`; confidence: 0.0-1.0; summary: <what code change likely caused the result>; evidence: <log/path>
```
