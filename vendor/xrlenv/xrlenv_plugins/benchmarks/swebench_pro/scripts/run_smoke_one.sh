#!/usr/bin/env bash
# scripts/run_smoke_one.sh — the quickest plumbing check: the gold-patch (oracle) run on ONE SWE-bench Pro task.
#
# Default task = the first id of subset_100_instance_ids.txt (a member of all three configurations:
# subset-100 ⊂ filtered ⊂ full). One image pull + one container + one verifier run: it proves the
# inputs (SWEBENCH_PRO_PARQUET, SWEBENCH_PRO_HARNESS), the cache, the cluster connection and the
# grading path end to end in a couple of minutes, before any warm-up or sweep.
#
#   --instance ID     run this instance instead
#   --index N         the N-th id (1-based) of the subset-100 manifest instead of the first
#   <anything else>   forwarded to run_full_sweep.sh (--list-green, --content-retries, --job-id,
#                     --jobs-dir, --skip-build-cache, run_oracle_sweep.py knobs)
#
# Same pipeline as the sweeps (run_full_sweep.sh --instances ID): build the task dir if missing →
# green-set gate → run_oracle_sweep.py. Job id swebench-pro-smoke-one-<timestamp>; artifacts under
# --jobs-dir (default ./tmp/sanity-checks/<job-id>/<instance>/verifier/{reward.json,stdout.log,…}).
# Exit 0 iff the task rewards > 0 — a FAIL here is a plumbing/corpus defect, never a model signal.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/subset_100_instance_ids.txt"

INSTANCE=""; INDEX=1; ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --instance)  INSTANCE="$2"; shift ;;
    --index)     INDEX="$2"; shift ;;
    *)           ARGS+=("$1") ;;
  esac
  shift
done
if [ -z "$INSTANCE" ]; then
  INSTANCE="$(grep -v '^#' "$MANIFEST" | grep -v '^[[:space:]]*$' | sed -n "${INDEX}p")"
  [ -n "$INSTANCE" ] || { echo "ERROR: no id at index $INDEX of $MANIFEST" >&2; exit 2; }
fi
case "$INSTANCE" in *,*) echo "ERROR: one task only (got a comma list); use run_full_sweep.sh --instances for several" >&2; exit 2 ;; esac

echo "==> smoke test, one task: $INSTANCE"
exec bash "$HERE/../run_full_sweep.sh" --instances "$INSTANCE" --max-workers 1 --job-id swebench-pro-smoke-one ${ARGS[@]+"${ARGS[@]}"}
