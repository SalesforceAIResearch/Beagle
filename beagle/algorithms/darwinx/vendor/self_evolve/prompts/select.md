# Pick the best commit

You are reviewing `{{ candidates | length }}` candidate commits produced by
`{{ kept_iterations }}` kept iteration(s) of `cursor-agent` work on
`monet_code`, all attempting to fix failing tasks
`{{ claimed_tasks | join("`, `") }}` on parent commit `{{ parent_short_sha }}`
(subset `{{ subset_label }}`).

Each candidate is the **tip** of one iteration (i.e. the commit you'd
get if you reset to it). Your job is to pick the single best candidate
— the one most likely to generalize beyond the claimed tasks — and
output its SHA in the strict format below. The orchestrator will
`git reset --hard <your pick>` and use it for the final eval.

## Iteration history

Parent (baseline) commit: `{{ parent_full_sha }}` — picking this means
"abandon this branch; no iteration was an improvement worth keeping".

{% for c in candidates %}
### Option {{ loop.index }}: tip `{{ c.short_sha }}`

- iteration: **{{ c.iteration }}** of {{ max_iters }}
- commits introduced in this iteration (chronological):{% for s in c.commits %}
  - `{{ s[:7] }}`{% endfor %}
- claimed-task mini-eval rewards:{% for t, r in c.claimed_rewards.items() %}
  - `{{ t }}`: **{{ "%.3f"|format(r) }}**{% endfor %}
- claimed wins: **{{ c.claimed_wins }}/{{ c.claimed_total }}**{% if c.claimed_pass_rates is defined and c.claimed_pass_rates %}
- claimed-task **pass-rates** (graded, k-sample — progress before a full flip):{% for t, pr in c.claimed_pass_rates.items() %}
  - `{{ t }}`: **{{ "%.2f"|format(pr) }}**{% endfor %}
- aggregate progress (sum of pass-rates): **{{ "%.2f"|format(c.claimed_progress) }}**{% endif %}
- canary tasks: `{{ c.canary_tasks | join(", ") if c.canary_tasks else "(none)" }}`
- failed canaries: `{{ c.failed_canary_tasks | join(", ") if c.failed_canary_tasks else "(none)" }}`
- canary summary: passed=**{{ c.canary_passed }}**, failures=**{{ c.canary_failures }}/{{ c.canary_total }}**, failure_rate=**{{ "%.3f"|format(c.canary_failure_rate) }}**
- sampled mini-eval net gain: **{{ c.mini_eval_net_gain }}** (claimed wins minus canary failures)
- mini-eval job dir: `{{ c.mini_eval_job_dir or "(missing)" }}`
- full SHA: `{{ c.full_sha }}`
{% endfor %}

## Selection criteria (in priority order)

1. **Expected full-eval utility** — prefer candidates with positive sampled
   mini-eval net gain and a strong claimed-task pass rate. The mini-eval is
   only a sample; do not require zero canary failures before allowing a
   candidate to reach full eval.
   - **Graded progress (when pass-rates are shown):** a claimed task is failing
     by definition, so a clean win is rare in one pipeline. Treat a *higher
     claimed-task pass-rate* as genuine progress even when it has not yet
     crossed to a full win (e.g. `0.00 → 0.33 → 0.67` across iterations is the
     fix converging, not a failure). Among candidates with the same number of
     clean wins, prefer the one with the higher aggregate pass-rate / progress:
     it is the better base for the next iteration to build on. Only fall back to
     the parent when *no* candidate shows either a win or a pass-rate above the
     parent's baseline on any claimed task.
2. **Canary risk** — treat `canary_passed=False` as risk evidence, not an
   automatic disqualifier. Prefer lower canary failure rates and fewer failed
   canaries, and be skeptical of candidates whose canary failures are broad,
   repeated, or larger than their claimed-task wins. A small number of canary
   failures can be worth carrying when the claimed-task gain is materially
   larger and the diff is plausibly general.
3. **Claimed-task pass rate** — when canary risk is comparable, the higher
   aggregate reward across the claimed tasks is better.
4. **Diff generality** — for two candidates with the same numeric
   profile, prefer the one whose diff is more general (smaller surface,
   no task-name literals, no copied verifier strings, no
   `if task_name == X` style narrowing).
5. **Diff minimality** — when generality is comparable, prefer the
   smaller diff (lower review/maintenance cost upstream).

Use the tools to inspect each candidate's diff:

```bash
# from {{ wt_dir }}/monet_code/
git show <short_sha>                    # full diff for one commit
git diff {{ parent_short_sha }} <short_sha>   # cumulative diff vs parent
```

Pick the parent commit `{{ parent_full_sha }}` only when every candidate is
net-negative or too risky to justify a full eval: for example, canary failures
outnumber claimed wins, canary failures are widespread across the sample, the
claimed-task reward AND pass-rate are not better than the parent, or the diff
looks overfit or unsafe. Do **not** pick the parent solely because
`canary_passed=False`, and do **not** abandon a candidate that made real
graded progress on a claimed task (a non-zero / rising pass-rate) just because
it has no clean win yet — that progress is exactly what the next iteration
builds on.

## Output format (STRICT — orchestrator parses this verbatim)

End your reply with **exactly** this fenced block, no extra lines after:

```
<<<SELECTED_COMMIT>>>
<40-char sha>
<<<END>>>
```

Any free-form rationale must come BEFORE the block. Keep it tight — a
paragraph or two explaining the trade-offs. The orchestrator will save
your full reply for the report.
