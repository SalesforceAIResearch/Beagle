#!/usr/bin/env bash
# Generate the BASELINE eval configs: EVERY onboarded experiment copy × EVERY benchmark.
# Baseline knobs: model=gpt-5.6-sol, effort=medium, max_turns=200.
#
# Configs  → experiments/configs/eval_baseline/{harness}-{version}_{bench}_{model}_{effort}_{turns}.yaml
# Raw runs → experiments/results/<same-stem>/  (baked into each config's run.dir)
#
# Usage:  bash experiments/scripts/generate_all_eval_configs.sh          # write configs
#         CHECK=1 bash experiments/scripts/generate_all_eval_configs.sh  # dry-run (write nothing)
set -euo pipefail

cd "$(dirname "$0")/../.."                       # → repo root
PY="${PY:-.venv/bin/python}"
GEN="experiments/scripts/generate_eval_configs.py"
MODEL="gpt-5.6-sol"; EFFORT="medium"; TURNS=200
PARALLELISM="${PARALLELISM:-16}"                 # force 16 for ALL benches (overrides deep-swe's 8)
CHECK_FLAG=""; [ "${CHECK:-0}" = "1" ] && CHECK_FLAG="--check"

common=(--model "$MODEL" --effort "$EFFORT" --max-turns "$TURNS" --parallelism "$PARALLELISM" $CHECK_FLAG)

# The generator defaults to the full cross-product of the canonical AGENTS / BENCHMARKS tables,
# so this stays correct when an agent is re-onboarded or a benchmark is added — naming them here
# is what silently dropped newly-onboarded entries before.
"$PY" "$GEN" "${common[@]}"

# --- narrowing (copies are <harness>-<version>; `--help` lists the current labels): ---
# "$PY" "$GEN" --agents monet-20260826 --benches swe-rebench "${common[@]}"

echo
echo "configs in: experiments/configs/eval_baseline/"
ls -1 experiments/configs/eval_baseline/ 2>/dev/null || true
