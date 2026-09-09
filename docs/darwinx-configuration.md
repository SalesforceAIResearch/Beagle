# Configure DarwinX

Beagle is the evaluation and evolution infrastructure. DarwinX is an evolution algorithm hosted by
Beagle and selected with:

```yaml
algorithm:
  name: darwinx
  hparams: {}
```

This guide maps the public configuration to the concepts in the
[DarwinX paper](https://arxiv.org/abs/2608.07545). The complete typed surface is
[`DarwinXConfig`](../beagle/algorithms/darwinx/config.py); an unknown `hparams` key fails when
`beagle evolve` builds the algorithm.

## Generate a starting config

First onboard the reference harnesses as described in the root README. Then generate versioned
DarwinX smokes whose source fields come from the matching manifests:

```bash
.venv/bin/python scripts/generate_evolution_config.py
.venv/bin/beagle evolve \
  --config tests/smoke/terminal_bench_2_1/opencode-1.18.16_darwinx_evolution_smoke2.yaml \
  --dry-run
```

The generated files are gitignored and safe to regenerate. Each contains two tasks, one proposal
attempt, one sample per evaluation stage, the selected public/internal provider profile, and Cursor
Agent as the evolver. Those settings test the plumbing; they are not statistically meaningful
experiments. Copy one under `experiments/configs/` before turning it into a persistent campaign.

## Read the config in five layers

```yaml
run:                         # Beagle infrastructure: outputs, runtime, concurrency
evolvee:                     # the versioned harness DarwinX changes (θ)
evolver:                     # the Editor that proposes source changes
algorithm:
  name: darwinx
  hparams:                   # DarwinX search and selection policy
data:                        # tasks that provide the evolution signal
```

### `run`: where evaluation executes

- `dir` and `name` identify the campaign. Reusing both resumes the same genealogy; changing `name`
  starts a new campaign.
- `runtime: local` uses the local Docker daemon. `xrlenv-cluster` uses Beagle's distributed rollout
  infrastructure.
- `parallelism` controls concurrent candidate-evaluation trials. Raise it only after the two-task
  path works and only within model-provider and container capacity.

These are Beagle infrastructure settings, not DarwinX method parameters.

### `evolvee`: what changes and what stays frozen

- `harness.source.repo` and `ref` identify the baseline harness. DarwinX writes candidate branches
  to this experiment copy.
- `harness.source.dir` is the local checkout from which DarwinX creates isolated worktrees.
- `model`, `provider`, `effort`, and agent budgets define the agent being measured.

Keep the evolvee model, provider, effort, task budgets, and baseline ref fixed within an experiment.
Otherwise a score change cannot be attributed to the evolved harness.

### `evolver`: who proposes changes

The current working public `Editor` integration is `cursor-agent`:

```yaml
evolver:
  harness: {name: cursor-agent}
  model: {name: auto}
```

Set a concrete model from `cursor-agent models` for a reproducible campaign. The registered
`claude-code` and `codex` adapters do not yet implement `edit()`, so they are not drop-in
alternatives in this release.

### `data`: where evolutionary pressure comes from

DarwinX currently trains on `data[0]`. `tasks` is the candidate pool it may claim, evaluate, and use
for preservation checks.

For a real campaign:

1. Measure the unevolved harness first with the same model and budgets.
2. Include tasks with observed headroom; an always-passing task cannot supply an improvement signal.
3. Retain some reliably passing tasks so regressions are visible.
4. Keep final validation tasks outside the evolution pool.

If every selected task passes at baseline, `no_change` is the expected result.

## Paper concepts and their knobs

### Branch evolution: propose bounded edits

```yaml
algorithm:
  hparams:
    total_steps: 1             # validate one node, then raise this campaign-wide budget
    merge_every: 0             # enable only after the archive has complementary parents
    max_loop_iters: 4
    n_failure_tasks: 2
```

- `max_loop_iters` is the maximum number of analyze/implement/review attempts made by **one
  pipeline**. It is not a generation or node budget.
- `n_failure_tasks` is how many currently failing tasks one pipeline claims. A larger pool can expose
  a shared failure mode, but it also makes the edit objective broader.

One pipeline produces one node. `total_steps` is the campaign-wide node budget (`1` when omitted).
It counts every non-root node, kept or discarded. Reuse `run.dir` / `run.name` to resume; changing
`run.name` starts a new campaign. `merge_every` needs `total_steps` greater than the cadence or
validation fails.

### Fitness and noise-aware confirmation

```yaml
algorithm:
  hparams:
    mini_eval_k_samples: 3
    fullset_metric: avg
```

- `mini_eval_k_samples` repeats each claimed and guard task during the inexpensive candidate screen.
  Use `1` only for plumbing; repeated samples expose flaky wins and regressions.
- `fullset_metric: avg` selects the driver's avg@k path. In this release its full-set `k` is the
  driver default of **5** and is not exposed by `DarwinXConfig`.
- `fullset_eval_n_attempts` applies to `fullset_metric: best`: it runs best-of-N and counts a task as
  solved if any attempt passes. It does **not** set the `k` of avg@k and is not a substitute for it.

Choose the mini-eval sample count from measured baseline variance and report the full-set metric
explicitly. Do not compare a best-of-N result with an avg@k baseline.

`subset_eval_n_attempts` exists in the typed surface but is not consumed by the current hosted
pipeline; do not rely on it.

### Preserve and extend: keep capabilities while adding one

```yaml
algorithm:
  hparams:
    guard_enabled: true
    guard_strict: false
    anti_cheat: true
    additive_scope: true
    preserve_extend: true
    require_extension: true
    routed_code: true
```

- `guard_enabled` evaluates parent-solved canaries alongside claimed tasks.
- `guard_strict` rejects any failed canary. Leave it `false` for noisy tasks unless repeated
  measurements show that a single failure is meaningful.
- How many parent-solved tasks are probed per node is a driver default (currently **1**). That
  count is not a typed `DarwinXConfig` field.
- `anti_cheat` rejects evaluation tampering; separate diff-scope checks handle narrow,
  task-specific edits.
- `additive_scope` limits destructive rewrites during ordinary extension nodes.
- `preserve_extend` adds the paper's preservation contract to the proposer context.
- `require_extension` prevents a child that adds no measured capability from being promoted.
- `routed_code` routes edits that touch shared engine paths through a specialist contract instead of
  admitting one monolithic change.

The generated smoke leaves the additional scope/preservation switches at their defaults so it
exercises the shortest path. Enable them deliberately for a paper-style campaign after the
baseline and canary sets are established.

### Node variants: what kind of edit a node may make

Every pipeline is drawn as one of three classes before it proposes anything, so this decides what an
edit is *allowed* to be, independent of how the result is later classified.

```yaml
algorithm:
  hparams:
    prune_enabled: true
    prune_rate: 0.2
    prune_min_lineage: 3
    consolidate_enabled: true
    consolidate_rate: 0.3
    consolidate_force_k: 4
    consolidate_on_bloat: true
    complexity_code_globs: "packages/opencode/src/**/*.ts"
    complexity_prompt_globs: "packages/opencode/src/session/prompt/**/*"
```

- **ADDITIVE** is the default and the only class active when both switches are off. It may only
  add, bound by the additive and extension contracts.
- **PRUNE** may only delete, and only what this campaign's lineage added.
- **CONSOLIDATE** may rewrite pre-evolve code. `consolidate_enabled` requires
  `complexity_code_globs` pointed at this evolvee's source. The globs above are OpenCode examples.

A scheduled consolidation outranks a prune draw. Config validation rejects rates that leave under
10% of nodes additive. Over a short campaign, zero prune nodes is ordinary: depth gate, then lottery.

### Population and archive selection

```yaml
algorithm:
  hparams:
    parent_strategy: mixed_high_score
    qd_archive: true
    total_steps: 15
    merge_every: 5
```

- `mixed_high_score` is the effective default. It mixes exploitation of cumulative lineage gain
  with breadth among less-expanded variants and requires no extra model call.
- `llm_first` asks a model to choose among archive cards and falls back to `mixed_high_score` on
  failure. It adds cost and another provider dependency.
- `qd_archive` retains a variant that cracks a previously unsolved claimed task even when it loses
  a guard, making that specialist available to recombination without treating it as the shipped
  best node.
- `hybrid_archive` and `archive_max_regressions` remain the bounded-regression archive path.
- `total_steps` is the campaign-wide node budget. It counts every non-root node, kept or discarded.
- `merge_every` attempts a recombination every N nodes (`0` never merges). Validation requires
  `total_steps > merge_every`. Concurrent workers can overshoot the cap; treat it as a coordination
  target unless you run one worker.

`archived` nodes can still be sampled as parents while being excluded from the shipped best node.

The vendored implementation also contains a regression-resolve pipeline, but the current Beagle
supervisor does not schedule it automatically. `regression_margin` controls when the ordinary gate
treats a measured drop as a regression; it does not enable that resolver. Estimate the margin from
baseline rerun variance, like the cross-benchmark margin below.

### Learning signals and shared memory

```yaml
algorithm:
  hparams:
    trace_qc: true
    collective_knowledge: true
    bestof2_contrast: true
```

- `trace_qc` extracts rule-based failure evidence from compatible retained trajectories. Confirm
  that the digest is non-empty for your evolvee's trajectory format before relying on it.
- `collective_knowledge` feeds lessons from earlier nodes back into later proposals.
- `bestof2_contrast` contrasts passing and failing attempts on variable tasks. It needs repeated
  samples; it has no useful signal when every task is always pass or always fail.

`trace_qc_llm`, `reasoned_verdict`, equivalence checking, and verifier-fitness fields add separate
model calls. Enable them only after their model/provider credentials have been validated. The
paper's teacher-derived signal is not exposed as a supported typed Beagle setting in this release.

The optional post-finalization verifier/scope gate is also off by default; set
`gate_enabled: true` and validate its provider before relying on its fields. The generated smoke
exercises the built-in canary/content/anti-cheat path, not that full LLM gate.

### Held-out and multi-benchmark protection

Cross-benchmark controls are advanced because their thresholds must come from measured baselines:

```yaml
algorithm:
  hparams:
    cross_bench_gate: true
    heldout_benchmark: swe-bench-verified
    heldout_tasks: "@/path/to/frozen-heldout-task-list.txt"
    heldout_k: 3
    cross_bench_margin: 0.05
```

Do not copy the margin above as a universal threshold: estimate it from repeated runs of the
unchanged baseline. A margin below natural rerun variation rejects healthy candidates; one above it
cannot detect regressions.

For multi-benchmark regression protection, `mixture_gate` requires a `mixture_spec` containing a
measured baseline and standard deviation for every member. It can veto a candidate that drops one
member and it records a normalized mixture score.

In the current hosted path, that mixture score does **not** replace the panel/search-evaluation
score used by parent selection: the vendor does not consume `node_score`. Do not present
`node_score: mixture` as a multi-benchmark search objective in this release. Configure the mixture
gate from calibration artifacts, not guessed weights.

### Deferred evaluation and fixed panels

`defer_node_full_eval: true` avoids a full benchmark evaluation for every candidate. It must be
paired with `fixed_eval_panel: true`; otherwise each node can receive a different denominator and
its score is not comparable. Beagle rejects that invalid combination.

```yaml
algorithm:
  hparams:
    defer_node_full_eval: true
    fixed_eval_panel: true
    eval_panel_size: 40
```

Use this only when a separate confirmation step will evaluate the selected candidates at the
reported metric and sample count.

## Suggested progression

### Plumbing smoke

The generated config uses:

```yaml
algorithm:
  hparams:
    max_loop_iters: 1
    n_failure_tasks: 1
    mini_eval_k_samples: 1
    fullset_eval_n_attempts: 1
    fullset_metric: best
    guard_enabled: true
    anti_cheat: true
```

Its only claim is that the source, evolver, model provider, container runtime, benchmark, and
candidate branch path work end to end. Omitting `total_steps` keeps the one-node smoke.

### First measured campaign

Start conservatively rather than maximizing every feature:

```yaml
algorithm:
  hparams:
    max_loop_iters: 4
    n_failure_tasks: 2
    mini_eval_k_samples: 3
    fullset_metric: avg
    parent_strategy: mixed_high_score
    guard_enabled: true
    guard_strict: false
    anti_cheat: true
    additive_scope: true
    preserve_extend: true
    require_extension: true
```

Then:

1. Dry-run and complete one pipeline.
2. Inspect whether claimed tasks had headroom and whether canaries were stable.
3. Raise `total_steps` to the measured compute budget and rerun the same campaign.
4. Evaluate the selected commit on untouched tasks with the same metric.
5. Only then add archive, LLM-judge, or cross-benchmark features one group at a time.

This progression keeps a failed run diagnosable and keeps each added source of cost attributable.

## Validation checklist

```bash
# Config shape, typed DarwinX knobs, and task resolution; no model calls or containers.
.venv/bin/beagle evolve \
  --config tests/smoke/terminal_bench_2_1/opencode-1.18.16_darwinx_evolution_smoke2.yaml \
  --dry-run

# One real pipeline; spends evolver and evolvee model credits.
.venv/bin/beagle evolve \
  --config tests/smoke/terminal_bench_2_1/opencode-1.18.16_darwinx_evolution_smoke2.yaml
```

Before scaling, confirm:

- the dry-run names the intended evolvee repo/ref, model, benchmark, task set, and runtime;
- the baseline and candidate both have measured trials rather than infrastructure errors;
- a candidate branch appears in the experiment-copy repository when an edit survives;
- `no_change` means the gate rejected/no edit survived, while `failed` means the pipeline broke;
- the final evaluation uses the same metric and sample count as the baseline.
