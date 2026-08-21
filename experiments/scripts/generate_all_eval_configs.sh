#!/usr/bin/env bash
# Generate the BASELINE eval configs: monet + opencode  ×  tb2.1 / deep-swe / swe-verified.
# Baseline knobs: model=gpt-5.6-sol, effort=medium, max_turns=200.
#
# Configs  → experiments/configs/eval_baseline/{harness}_{bench}_{model}_{effort}_{turns}.yaml
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

# --- one command per config (agent × benchmark) — comment any line to skip it ---
# monet
"$PY" "$GEN" --agents monet    --benches terminal_bench_2_1 "${common[@]}"
"$PY" "$GEN" --agents monet    --benches deep-swe           "${common[@]}"
"$PY" "$GEN" --agents monet    --benches swe-bench-verified "${common[@]}"
# opencode
"$PY" "$GEN" --agents opencode --benches terminal_bench_2_1 "${common[@]}"
"$PY" "$GEN" --agents opencode --benches deep-swe           "${common[@]}"
"$PY" "$GEN" --agents opencode --benches swe-bench-verified "${common[@]}"

# --- equivalent single call (the generator takes the full cross-product): ---
# "$PY" "$GEN" --agents monet opencode \
#     --benches terminal_bench_2_1 deep-swe swe-bench-verified "${common[@]}"

echo
echo "configs in: experiments/configs/eval_baseline/"
ls -1 experiments/configs/eval_baseline/ 2>/dev/null || true
