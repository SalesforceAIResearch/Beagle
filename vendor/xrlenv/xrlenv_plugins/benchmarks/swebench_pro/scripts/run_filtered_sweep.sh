#!/usr/bin/env bash
# scripts/run_filtered_sweep.sh — the gold-patch (oracle) sweep over the quality-FILTERED SWE-bench Pro set.
#
# The selection = filtered_instance_ids.txt: the 478 of the 731 public instances our filter keeps
# (drop reasons per instance in filter_report.json; see README "Three configurations").
# Images: warm them first with build_plan_filtered.yaml (478 images, ~693 GB compressed):
#   xrlenv build apply --plan xrlenv_plugins/benchmarks/swebench_pro/scripts/build_plan_filtered.yaml
#
# This is run_full_sweep.sh with the selection pinned: every flag it accepts is forwarded
# (--max-workers, --content-retries, --job-id, --jobs-dir, --skip-build-cache, --list-green, run_oracle_sweep.py knobs).
# Default job id: swebench-pro-filtered-sweep. The agent is harbor's OracleAgent (solution/solve.sh = the gold patch) —
# an oracle FAIL is a corpus/plumbing defect, never a model signal.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../run_full_sweep.sh" --filtered "$@"
