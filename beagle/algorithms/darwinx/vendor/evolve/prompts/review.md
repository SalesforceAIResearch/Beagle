# Review, test, fix, and commit

Review the diff in `{{ wt_dir }}/monet_code/`, run focused checks, fix bugs you
introduced, and commit. Keep scope tied to the plan and avoid broad provider/API
integration suites unless the diff requires them.
Your primary job is to catch bugs introduced by the implementation step.
Use at most 1 self-debug iteration unless a tiny follow-up fix is obvious.
Do not spend the review budget repairing unrelated test infrastructure or broad-suite flakiness.
Run full `npm test` only when the change is broad or focused evidence suggests wider risk.

Context: pipeline `{{ pipeline_id }}`, iteration `{{ iteration }} / {{ max_iters }}`,
branch `{{ monet_branch }}`, base commit `{{ parent_commit }}`, claimed tasks
`{{ claimed_tasks | join(", ") }}`.

Steps:
1. Inspect `git -C {{ wt_dir }}/monet_code diff {{ parent_commit }}..HEAD`.
2. Run focused offline tests/build checks that exercise the changed behavior.
3. Fix implementation bugs exposed by those checks.
4. Commit with:

   ```
   <subject line under 72 chars, imperative mood>

   <one-paragraph "why" — what general problem this solves>

   Iteration {{ iteration }} of self-evolve pipeline {{ pipeline_id }}.
   Targeted failing tasks: {{ claimed_tasks | join(", ") }}.
   ```

Do not introduce task-name literals, copied verifier strings, trial ids, or
task-specific path gates in the diff.

In your final reply, include:

```markdown
## Generalization Audit

### Why the fix is general
<one paragraph>

### Validation
<commands/checks run and outcome>

## Shared Experience Updates
- task: `<task>`; kind: improved|regressed; commit: `<sha>`; confidence: 0.0-1.0; summary: <what code change likely caused the result>; evidence: <log/path>
```
