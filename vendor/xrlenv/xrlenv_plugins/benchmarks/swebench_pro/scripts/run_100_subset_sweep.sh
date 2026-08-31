#!/usr/bin/env bash
# scripts/run_100_subset_sweep.sh — the gold-patch (oracle) sweep over the 100-task sample of SWE-bench Pro.
#
# The selection = subset_100_instance_ids.txt: 100 instances drawn from the quality-filtered set (478) and
# spread over all 11 repositories proportionally to their kept counts (every repo ≥ 1; policy random,
# seed 0 — per-repo counts and image sizes in subset_100.json; regenerate with
#   .venv/bin/python -m xrlenv_plugins.benchmarks.swebench_pro.scripts.sample_subset            # defaults = this subset
# Images: warm them first with build_plan_subset_100.yaml (100 images, ~144 GB compressed):
#   xrlenv build apply --plan xrlenv_plugins/benchmarks/swebench_pro/scripts/build_plan_subset_100.yaml
#
# This is run_full_sweep.sh with the selection pinned: every flag it accepts is forwarded
# (--max-workers, --content-retries, --job-id, --jobs-dir, --skip-build-cache, --list-green, run_oracle_sweep.py knobs).
# Default job id: swebench-pro-subset100-sweep. The agent is harbor's OracleAgent (solution/solve.sh = the gold patch) —
# an oracle FAIL is a corpus/plumbing defect, never a model signal.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../run_full_sweep.sh" --subset-100 "$@"
