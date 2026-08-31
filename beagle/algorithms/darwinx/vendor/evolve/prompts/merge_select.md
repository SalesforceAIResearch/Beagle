# Select two self-evolve nodes to merge

You are choosing a pair of self-evolve graph nodes for the merger pipeline.

Pipeline: `{{ pipeline_id }}`

## Candidate Nodes

```json
{{ candidates_json }}
```

## Allowed Unordered Pairs

```json
{{ allowed_pairs_json }}
```

Allowed pairs already exclude nodes whose solved-task sets are identical or
where one node contributes no solved task beyond the other. The two selected
nodes must therefore contribute different solved tasks.

## Pair Risk / Complementarity Summaries

```json
{{ pair_summaries_json }}
```

These summaries estimate whether a pair is likely to preserve both parents'
wins. Prefer high `expected_utility`, explicit `complementary_wins`, few
`shared_failures`, few `fragile_parent_wins`, and no
`prior_failed_similar_merges`.

## Solved Tasks Already Covered By In-Flight Merge Picks

```json
{{ reserved_solved_tasks_json }}
```

Choose the ordered pair with the best expected chance of producing a merged
child node that outperforms both parents. The first id is the primary/base
parent. Prefer a stable, less-regressive base and use the other parent as the
specialist whose unique wins should be transplanted. Prefer complementary
solved-task sets and avoid pairs whose unsolved-task lists suggest the same
remaining failure mode or whose parent wins have repeatedly regressed in
previous merge/resolver attempts. When scores are tied, prefer pairs that add
solved tasks not already covered above.

Return only this block:

```text
<<<MERGE_PAIR>>>
node_id_a node_id_b
<<<END>>>
```
