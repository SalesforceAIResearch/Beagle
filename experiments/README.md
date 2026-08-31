# experiments/ — evaluation workspace

This directory holds parameterized baseline configs, raw evaluation runs, and local
analysis/recovery tools. The reusable evaluation examples remain under
`examples/evaluation/`; this workspace is for concrete experiment sweeps and their output.

```
experiments/
  configs/
    eval_baseline/                 # generated baseline YAML configs
  results/
    <config-stem>-<timestamp>/     # one run and its native artifacts
  scripts/
    generate_eval_configs.py       # parameterized baseline config generator
    generate_all_eval_configs.sh   # wrapper for the current six-run matrix
    dashboard.py                   # Streamlit results dashboard
    results_data.py                # dashboard discovery and summary cache
    grade_run.py                   # recover grading for an interrupted two-phase run
```

Config filenames and `run.name` describe the sweep point:
`{harness}_{bench}_{model}_{effort}_{max_turns}.yaml`. Benchmark short names currently
include `terminal_bench_2_1 → tb21`, `deep-swe → deepswe`, and
`swe-bench-verified → swebench_verified`.

Evaluations timestamp that base name by default, producing a fresh directory such as
`monet_tb21_gpt-5.6-sol_medium_200-20260825-102200`. The `run.json` inside it also
records beagle's collision-safe run ID and config hash.

## Generate baseline configs

The current baseline is **monet + opencode × Terminal-Bench 2.1 / DeepSWE /
SWE-bench Verified**, using `gpt-5.6-sol`, `medium` effort, and `200` turns.

```bash
# all six baseline configs
bash experiments/scripts/generate_all_eval_configs.sh
CHECK=1 bash experiments/scripts/generate_all_eval_configs.sh    # dry-run, write nothing
```

Or drive the generator directly — every knob is a flag:

```bash
# one config
.venv/bin/python experiments/scripts/generate_eval_configs.py --agents monet --benches deep-swe

# a different sweep (new model/effort/turns → new filenames, no collision)
.venv/bin/python experiments/scripts/generate_eval_configs.py \
    --agents monet opencode --benches terminal_bench_2_1 deep-swe swe-bench-verified \
    --model gpt-5.6-sol --effort medium --max-turns 200

.venv/bin/python experiments/scripts/generate_eval_configs.py --help   # all flags
```

Important flags include `--agents`, `--benches`, `--model`, `--effort`,
`--max-turns`, `--parallelism`, `--timeout`, `--retry-infra`, `--runtime`,
`--provider`, `--out`, `--results`, `--manifest-dir`, and `--check`.

Agent arguments, benchmark dataset/split settings, per-benchmark parallelism, forwarded
credentials, and the version-to-manifest join come from the canonical
`scripts/generate_eval_configs.py`. Onboard each agent version once under
`.beagle/agents/`; the experiment generator reuses it.

## Run

```bash
source .venv/bin/activate
beagle evaluate --config experiments/configs/eval_baseline/monet_tb21_gpt-5.6-sol_medium_200.yaml
```

The config's `run.dir` selects `experiments/results/` as the results root and `run.name`
selects the directory's base name. Each invocation appends a local timestamp by default:

```
experiments/results/<config-stem>-<YYYYmmdd-HHMMSS>/
  run.json                         # beagle run record and benchmark summaries
  <benchmark>/                     # benchmark-native job tree
    <trial>/                       # native trial artifacts
      agent/trajectory.json        # canonical ATIF trajectory
      result.json
      ...                          # harness-specific agent/verifier/artifact files
  summary.json                     # generated on demand by the dashboard
```

The benchmark subtree intentionally follows its native harness contract; do not expect
all benchmarks to contain exactly the same files. `summary.json` is a disposable,
git-ignored dashboard cache rather than part of the evaluation record.

To continue a specific interrupted run, pass its directory explicitly:

```bash
beagle evaluate \
  --config experiments/configs/eval_baseline/monet_tb21_gpt-5.6-sol_medium_200.yaml \
  --resume --run-dir experiments/results/<config-stem>-<timestamp>
```

For browsing results or recovering an interrupted SWE-bench grading phase, see
[`scripts/README.md`](scripts/README.md).

## Prereqs
- The gateway relay must be running and the config's forwarded gateway credentials must
  be available. Relay scripts live under `scripts/gateway/`; supported model/effort
  combinations are documented in `notes/gateway-models.md`.
- DeepSWE requires `uv pip install -e '.[deep-swe]'`.
- Agents must be onboarded at the versions selected by the canonical generator. Missing
  versions under `.beagle/agents/` are skipped during generation.
