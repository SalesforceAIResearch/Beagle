# Evolution quick start — run one DarwinX pipeline

This example evolves the **OpenCode harness** with **Cursor Agent** on one
Terminal-Bench 2.1 task. DarwinX scores the baseline, asks the evolver to edit the harness, scores
the candidate, applies preservation and anti-cheat checks, and records the best surviving harness.

Each generated config deliberately uses two tasks, one proposal attempt, and one sample per
evaluation stage. It is an end-to-end plumbing smoke, not a benchmark result.

## 1. Prepare the example

From the Beagle repository root:

```bash
bash scripts/install.sh
cp .env.example .env

# In .env, set YOUR_ORG, GH_TOKEN, and the key used by the example:
# OPENAI_API_KEY=...

docker info

# Download only the benchmark used by this smoke.
.venv/bin/python scripts/populate_benchmarks_cache.py \
  --benchmark terminal_bench_2_1

# Create the reference harness experiment copies and local checkouts.
bash scripts/onboard_all_agents.sh
```

Onboarding writes versioned manifests under `.beagle/agents/`. Each pins the repository, baseline
commit, clone credential, and local checkout DarwinX will evolve.

Cursor Agent is currently the working public **evolver** integration. Confirm that the CLI Beagle
invokes is installed and authenticated:

```bash
cursor-agent --version
cursor-agent models
```

The smoke uses Cursor's `auto` model. You can pin a model in the generated YAML after the first
dry-run.

## 2. Generate the evolution config

```bash
.venv/bin/python scripts/generate_evolution_config.py
# wrote tests/smoke/<bench>/<harness>-<version>_darwinx_evolution_smoke2.yaml
```

The generator reuses the evaluation generator's harness/benchmark matrix, provider profile, and
seeded task samples. It fills `evolvee.harness.source` from each matching onboard manifest. The
resulting smoke configs are gitignored and safe to regenerate.

## 3. Preview, then launch

```bash
# Resolve the source, models, tasks, runtime, and DarwinX parameters. No model calls or containers.
.venv/bin/beagle evolve \
  --config tests/smoke/terminal_bench_2_1/opencode-1.18.16_darwinx_evolution_smoke2.yaml \
  --dry-run

# Launch one DarwinX pipeline. This uses evolver and evolvee model credits.
.venv/bin/beagle evolve \
  --config tests/smoke/terminal_bench_2_1/opencode-1.18.16_darwinx_evolution_smoke2.yaml
```

The dry-run validates the config shape and DarwinX knobs and resolves the tasks. It cannot validate
the local checkout, evolver authentication, provider credentials, or container egress; the real
two-task launch checks those seams.

The command above is one generated example. Select the corresponding `<harness>-<version>` file to
smoke another onboarded harness. If both selected tasks pass at baseline, DarwinX may correctly
return `no_change`; that is not an infrastructure failure.

## 4. Understand and customize the YAML

The generated config keeps each concern in one block:

```yaml
run:                         # Beagle runtime, output path, and concurrency
evolvee:                     # versioned OpenCode harness + frozen model/provider
evolver:                     # Cursor Agent and its model
algorithm:
  name: darwinx
  hparams:                   # DarwinX search, measurement, and gates
data:                        # benchmark and evolution task pool
```

Start with [Configure DarwinX](../../docs/darwinx-configuration.md). It maps `hparams` to the
paper's branch evolution, noise-aware fitness, preserve-and-extend, population selection, learning
signals, and held-out protection; it also identifies settings that must be calibrated together.

To use Anthropic for the **evolvee**, change its model, provider, and forwarded key together:

```yaml
evolvee:
  model: {name: claude-opus-4-8}
  provider: {type: direct, name: anthropic}
  forward_env: [ANTHROPIC_API_KEY]
```

This changes OpenCode's model calls. It does not change the evolver: the current public DarwinX
integration still uses Cursor Agent to propose harness edits.

The Python `Trainer` version remains available for users who need programmatic composition:

```bash
.venv/bin/python examples/evolution/quick_start_inline.py --runtime local --dry-run
```

It is an advanced alternative to the YAML CLI, not an additional quick-start step.

## What the run produces

Artifacts land under `<run.dir>/<run.name>/`; the generated example above uses
`tmp/opencode-1.18.16-tb21-darwinx_evolution_smoke2/`:

```text
tmp/opencode-1.18.16-tb21-darwinx_evolution_smoke2/
├── _evals/                         benchmark-native rollout artifacts
├── <campaign-state>/
│   ├── state.db                    genealogy and node scores
│   ├── nodes/                      accepted/archived node evidence
│   └── pipelines/                  proposal and evaluation logs
└── worktrees/                      isolated candidate worktrees
```

A kept candidate is also pushed to your experiment copy as
`evolve/<parent>__<pipeline>`. `no_change` means no proposed edit survived; `failed` means the
pipeline itself broke.

## Move beyond the smoke

1. Evaluate the unevolved harness at the same model and budgets.
2. Replace the one smoke task with a measured pool containing both headroom and stable canaries.
3. Increase sample counts to match the noise of that pool.
4. Raise `algorithm.hparams.total_steps` and rerun to grow the same campaign genealogy.
5. Evaluate the selected commit on tasks DarwinX never saw.

Do not increase concurrency or enable every advanced gate at once. Add one parameter group at a
time so cost changes and failures remain attributable.
