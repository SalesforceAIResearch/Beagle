# Comprehensive campaign report for `{{ campaign }}`

You're being asked to synthesize a comprehensive markdown report covering
the entire **{{ campaign }}** self-evolve campaign so far. This is
distinct from the per-node `learnings.md` files (those are local to one
branch); the report you're writing covers the whole tree.

## Inputs

- Campaign digest (machine-readable JSON dump of the SQLite tree, with
  every node's score, parent, delta vs parent / root, status, resolved
  tasks, PR URL, and pointers to per-node markdown files):
  `{{ digest_path }}`
- All per-node markdown lives under `{{ campaign_dir }}/nodes/<node_id>/`.
  Specifically `works.md`, `effort.md`, and `learnings.md` (when present).
- All pipeline artifacts (cursor-agent stream-json logs, prompts,
  per-iteration logs) live under `{{ campaign_dir }}/pipelines/<pid>/`.

You are running in **read-only plan mode** (`--mode plan`). Use shell tools
to grep/read these inputs as needed.

## Aggregate stats (pre-computed)

- Total nodes attempted: `{{ stats.total_nodes }}`
- Completed nodes (final score recorded): `{{ stats.completed }}`
- Best score so far: `{{ stats.best_score }}` (root score: `{{ stats.root_score }}`,
  uplift: `{{ stats.uplift }}`)
- Biggest single-step uplift: `{{ stats.biggest_step }}`
- Total cursor-agent token cost: `{{ stats.total_tokens }}`
  (input: `{{ stats.input_tokens }}`, output: `{{ stats.output_tokens }}`,
   cache read: `{{ stats.cache_read_tokens }}`)
- Total wall-clock across all pipelines: `{{ stats.wall_clock_human }}`

## What to write

Produce a single comprehensive markdown report. **Reply with the markdown
only** — no surrounding prose. The orchestrator saves your reply verbatim
to `{{ output_path }}`.

Structure:

```markdown
# {{ campaign }} — campaign report ({{ stats.report_generated_at }})

## TL;DR

<3-4 sentences: where the campaign started, where it is now, biggest win,
biggest pattern.>

## Best results

| Rank | Node | Score | Δ vs parent | Δ vs root | Resolved | PR |
|------|------|-------|-------------|-----------|----------|-----|
| 1 | `evolve/abc__pid` | 0.600 | +0.083 | +0.083 | task1, task2 | #42 |
| ... |

(Top {{ stats.top_n }} from the digest's `top_nodes` array; one row per node.)

## What worked

<3–6 bullet points naming recurring patterns across the highest-scoring
nodes. Cross-reference at least one `nodes/<id>/learnings.md` per
pattern.>

## What didn't work

<3–6 bullets on `no_change` / reverted-iteration patterns. Look at
`nodes/<id>/effort.md` for the cluster of "Generalization-guard rejected
iteration" entries.>

## Root-cause clusters

<group all monet_code bugs found by the campaign into named buckets
(e.g. "tool-call truncation recovery", "permission classifier overreach",
"missed self-verification"). For each bucket, list which nodes attacked
it, the typical fix, and whether it generalized.>

## Cost & efficiency

- Tokens per uplift point: <X>
- Wall-clock per uplift point: <Y>
- Best ROI configuration: <model + iteration count + parent-strategy that
  gave most score-per-token>

## Recommendations for next campaign

1. <concrete next experiment, with the parent node it should branch from>
2. <prompt refinement to try based on patterns above>
3. ...

## Appendix — full node inventory

<one-line summary per node, sorted by created_at, format:
`{node_id} {status:>10} {score:>5.3f} ({delta:+.3f}) {branch}`>
```
