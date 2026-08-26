# Incorporating trace_analyzer into the evolving systems

Both evolving systems — `self_evolve/` (cursor meta-agent improves `monet_code`)
and `atelier/` (evolutionary search) — currently diagnose failures by **reading
raw trajectories** and guessing the root cause. `trace_analyzer` replaces that
vague step with **structured, evidence-cited QC**: a fixed taxonomy
(`tool_error`, `incomplete_run`, `early_truncation`, `behavioral_loop`,
`premature_completion`, `instruction_not_followed`, `fabricated_facts`) with the
exact `message_index` + evidence for each finding.

Notably, `self_evolve/prompts/analyze.md`'s "high-leverage scaffold directions"
are already this taxonomy in prose (verification discipline → `premature_completion`,
tool-use reliability → `tool_error`, budget/thrash → `behavioral_loop`,
timeout-robust → `incomplete_run`). So this closes the loop: the manual
`analysis/evolve/monet_failure_analysis.md` that *shaped* that prompt becomes an
automatic per-iteration input.

## Format note (why this works on monet runs)

The reusable engine reads **monet stream-json `*.trajectory.jsonl`**, which is
what monet writes (`agents/monet/harbor_agent.py`: `SUPPORTS_ATIF = False`).

This is the format in practice, verified empirically: across **all 14 runs in
`results/runs/` (2,231 trajectories, 4 benchmark families — swe-bench-*,
terminal-bench, harbor-seta, harbor-turing), 100% are stream-json `.jsonl` and
**0** are ATIF `trajectory.json` or `transcript.md`. They all live at
`<run>/raw/<task>.trajectory.jsonl`, and the analyzer parses every one
(`source=monet`). So ATIF is not a real concern here — an `atif` normalizer
would only matter if a future campaign evaluated an ATIF-native (non-monet)
agent.

The digest is still defensive — a trajectory it can't read is **recorded and
counted, never fatal** — so an old/missing sidecar degrades gracefully instead
of breaking analyze; the self_evolve adapter finds the sidecar by globbing
`<eval_dir>/**/<task>.trajectory.jsonl`, matching the `raw/` layout above.

## The shared surface

`trace_analyzer/digest.py` is what both systems call:

```python
from trace_analyzer.digest import digest_paths
d = digest_paths([(task_id, traj_path), ...])   # rule-only, offline by default
block = d.render_markdown()                       # paste into the analyze prompt
d.to_dict()                                       # structured artifact
```

Pass `llm=<client>` to also run the semantic proposers. Everything below is just
*where each system calls this and injects the block*.

---

## Phase 1 — evidence block into `analyze` (DONE)

Feed a QC digest into each system's analyze step. Deterministic, offline, zero
API cost → safe to run every iteration.

- **Shared:** `trace_analyzer/digest.py` (`digest_paths` → `render_markdown`).
- **self_evolve:** `self_evolve/trace_qc.py` locates each claimed task's
  `*.trajectory.jsonl` under the eval dir and builds the block;
  `pipeline._prompt_context` exposes it as `trace_qc_summary`;
  `prompts/analyze.md` (and later `regression_analyze.md` / `merge_*`) renders a
  `{{ trace_qc_summary }}` section.
- **atelier:** `atelier/trace_analyzer.py`'s per-trial digest gains a
  "trace_analyzer QC" section sourced from the shared engine when a `.jsonl` is
  present, so its existing `analyze`-feeding digests carry the real taxonomy.

Framing is **advisory** in the prompt — "candidate symptoms, not verdicts; empty
≠ clean; confirm against originals" — so it guides without inducing overfitting
(the prompt already warns against overfitting to task specifics).

## Phase 2 — QC as a selection + progress signal

- **Selection:** cluster the failing set by category and pick a *representative*
  claimed-task set (target a generalisable pattern, not a one-off) in
  `self_evolve/parent_selection.py` / `adaptive_subset.py`.
- **Progress:** track per-iteration category counts as a **denser fitness signal**
  than binary reward (accuracy is sparse/noisy) — "did `premature_completion`
  drop after this change?"

## Phase 3 — QC-diff for regression / merge gating

`regression_pipeline` / `merge_pipeline` already re-run evals. Compute a
**before/after QC diff** to confirm a candidate removed the targeted category
without introducing a new one (a verification-gate fix should cut
`premature_completion` without raising `behavioral_loop`). Optionally reject a
candidate that reduces accuracy *or* regresses a targeted category.

## Phase 4 — LLM proposers for the residue

For "wrong-but-clean" failures (rule QC empty, e.g. a fix validated only on the
literal example), run `digest_paths(..., llm=client)` on just the handful of
claimed tasks to add the semantic proposers (`premature_completion_llm`,
`instruction_not_followed_llm`, `fabricated_facts_llm`). Gate it (costs API
calls; the meta-agent is itself an LLM, so the value is a *consistent fixed
taxonomy*, not raw capability). Pairs naturally with Phase 3's diff.
