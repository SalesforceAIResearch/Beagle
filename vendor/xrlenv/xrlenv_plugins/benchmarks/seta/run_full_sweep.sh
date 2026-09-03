#!/usr/bin/env bash
# run_full_sweep.sh — launch the seta-env "green" oracle sweep (the CI gate).
#
# The green set = all present seta tasks MINUS black_list.txt (the upstream tasks
# whose Dockerfiles never built — no image in the registry). The exclusion lives
# in ONE place: black_list.txt, applied by `run_oracle_sweep.py --all` (Python).
# This script does NOT re-parse the blacklist — it delegates to `--all`, so there
# is a single source of truth and a single exclusion code path.
#
# An oracle FAIL here is a corpus/plumbing defect (reward ceiling 0 → poison for
# RL), not a model signal — inspect the per-trial verifier output.
#
# seta tasks ship a Dockerfile (not a prebuilt docker_image); the cluster resolves
# each task to <registry>/seta-env/<id>:main via the xrlenv_image_template kwarg,
# which run_oracle_sweep.py composes from XRLENV_PRIVATE_REGISTRY_HOST/PORT in .env.
#
# Typical use:
#     bash xrlenv_plugins/benchmarks/seta/run_full_sweep.sh                    # full green set
#     bash xrlenv_plugins/benchmarks/seta/run_full_sweep.sh --max-workers 32   # override concurrency
#
# This script:
#   1. (re)builds the cache with `build_cache.py --stage all` — clone into the
#      seta-env/ shard + write the DinD sysbox routing markers (§1.1 in README).
#      Idempotent: on an already-populated shard the clone is skipped and only the
#      markers re-apply. So the DinD fix is honored here, automatically — no
#      separate `--stage sysbox` step for the operator to remember. Then
#   2. runs `run_oracle_sweep.py --all` (green set = present − black_list.txt),
#      content-retrying reward-0 flakes.
#
# Flags (the interface — run knobs are FLAGS, not env vars: a stale exported
# SKIP_BUILD/LIST_GREEN must never silently turn a real sweep into a no-op.
# XRLENV_* deployment config still comes from .env):
#   --max-workers N       trial concurrency (default: 16)
#   --content-retries N   reward-0 re-run rounds (default: 0)
#   --job-id LABEL        run label under tmp/ (default: seta-full-sweep; ts appended)
#   --jobs-dir DIR        per-trial artifact root (default: ./tmp/sanity-checks)
#   --skip-build-cache    skip step 1 (cache already built this session)
#   --list-green          print the green set (present − black_list.txt) and exit
#   <anything else>       forwarded to run_oracle_sweep.py
# XRLENV_BENCHMARK_CACHE (cache ROOT, default /path/to/xrlenv_benchmark_cache) comes from env / .env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ── flags (the interface) + passthrough for run_oracle_sweep.py ───────────────
LIST_GREEN=0
SKIP_BUILD=0
MAX_WORKERS=16
JOB_ID="seta-full-sweep"
CONTENT_RETRIES=0
JOBS_DIR="./tmp/sanity-checks"
PASS_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --list-green)        LIST_GREEN=1 ;;
    --skip-build-cache)  SKIP_BUILD=1 ;;
    --max-workers)       MAX_WORKERS="$2"; shift ;;
    --content-retries)   CONTENT_RETRIES="$2"; shift ;;
    --job-id)            JOB_ID="$2"; shift ;;
    --jobs-dir)          JOBS_DIR="$2"; shift ;;
    --dest|--dest=*|--cache|--cache=*)
      # H9: the cache root is XRLENV_BENCHMARK_CACHE ONLY. A pass-through --dest reaches the
      # evaluator (argparse last-wins) and would gate one cache but evaluate another.
      echo "ERROR: --dest/--cache is not accepted here — set XRLENV_BENCHMARK_CACHE instead" >&2
      echo "       (a pass-through cache override would gate one cache and evaluate another, audit H9)." >&2
      exit 1 ;;
    *)                   PASS_ARGS+=("$1") ;;
  esac
  shift
done
# audit Low: --list-green is a READ-ONLY query of the already-built cache — it must NOT
# (re)build (build_cache can need registry creds / mutate the cache). Imply --skip-build-cache.
if [ "$LIST_GREEN" = 1 ]; then SKIP_BUILD=1; fi

JOB_ID="${JOB_ID}-$(date +%Y-%m-%d_%H-%M-%S)"

# CP creds (host / token) AND the cache ROOT come from .env. Source it FIRST so the ONE
# resolved XRLENV_BENCHMARK_CACHE below drives EVERY stage — shell green-set enumeration AND
# the Python build/eval children (audit H9: otherwise SHARD is fixed pre-source while the
# children inherit the post-source value, so a direct run could gate one cache and execute
# another). A pure --list-green read needs no CP creds, and sourcing .env there would clobber
# a caller-provided cache (a unit test's fixture shard), so skip .env entirely when listing.
if [ "$LIST_GREEN" != 1 ]; then
  set +u; set -a; source ./.env 2>/dev/null || true; set +a; set -u
fi
if [ -z "${XRLENV_BENCHMARK_CACHE:-}" ]; then
  echo "ERROR: XRLENV_BENCHMARK_CACHE is not set. Set it (in .env or the environment) to your" >&2
  echo "       benchmark cache ROOT — the shared directory build_cache.py populates, e.g." >&2
  echo "       XRLENV_BENCHMARK_CACHE=/path/to/xrlenv_benchmark_cache (see .env.example)." >&2
  exit 1
fi
export XRLENV_BENCHMARK_CACHE

# audit (cache rename): the old XRLENV_HARBOR_CACHE env var and the old
# .../xrlenv_harbor_cache path are RETIRED. Fail loud so a stale env can't run against the
# wrong (moved/absent) cache and give unreliable results — use XRLENV_BENCHMARK_CACHE +
# /path/to/xrlenv_benchmark_cache. (The Python entrypoints guard this too.)
if [ -n "${XRLENV_HARBOR_CACHE+x}" ]; then
  echo "ERROR: XRLENV_HARBOR_CACHE is retired — unset it and set XRLENV_BENCHMARK_CACHE=/path/to/xrlenv_benchmark_cache" >&2
  exit 1
fi
case "$XRLENV_BENCHMARK_CACHE" in
  *xrlenv_harbor_cache*)
    echo "ERROR: retired cache path '$XRLENV_BENCHMARK_CACHE' — use /path/to/xrlenv_benchmark_cache" >&2
    exit 1 ;;
esac
SHARD="$XRLENV_BENCHMARK_CACHE/seta-env"
SETA="xrlenv_plugins/benchmarks/seta"
PY=".venv/bin/python"
# ── 1) (re)build the cache ────────────────────────────────────────────────────
if [ "$SKIP_BUILD" = 1 ]; then
  echo "==> --skip-build-cache — skipping build_cache"
else
  echo "==> build_cache --stage all  (clone seta-env into the cache shard)"
  "$PY" "$SETA/build_cache.py" --stage all
fi

if [ ! -d "$SHARD" ]; then
  echo "ERROR: shard not found at $SHARD — did build_cache populate?" >&2
  exit 1
fi

# --list-green (xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py sample mode): print the green set
# (present − black_list.txt) and exit. seta's blacklist is the single source; the
# first whitespace token of each non-comment line is the excluded task id.
#
# CONTRACT — NO fixed-count completeness gate (unlike deep_swe/lhtb/terminal_bench_2_1/
# terminalworld). seta's corpus is DYNAMIC BY DESIGN: green set = present − black_list.txt,
# with no pinned catalog size (build_cache clones upstream; a re-populate legitimately
# grows/shrinks the present set). Two invariants instead (audit M5):
#   * list vs execution parity — run_oracle_sweep's OracleAgent runs solution/solve.sh, and
#     _locate_task_dir / --all both require it; a stray/partial dir lacking solution/solve.sh
#     would be sampled then crash at setup, so only list tasks that carry solution/solve.sh.
#   * nonzero floor — an EMPTY green set is never legitimate (un-built shard, wrong
#     XRLENV_BENCHMARK_CACHE, or a blacklist that ate every task); fail rather than hand a
#     --list-green consumer a "0 tasks → nothing to run → green" no-op.
if [ "$LIST_GREEN" = 1 ]; then
  mapfile -t BLACK < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SETA/black_list.txt" 2>/dev/null)
  mapfile -t PRESENT < <(find "$SHARD" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
  GREEN=()
  for t in "${PRESENT[@]}"; do
    skip=0; for b in "${BLACK[@]:-}"; do [ "$t" = "$b" ] && { skip=1; break; }; done
    [ "$skip" -eq 1 ] && continue
    [ -f "$SHARD/$t/solution/solve.sh" ] || continue   # list vs execution parity (see above)
    GREEN+=("$t")
  done
  if [ "${#GREEN[@]}" -eq 0 ]; then
    echo "ERROR: green set is EMPTY (present=${#PRESENT[@]}, blacklisted=${#BLACK[@]}) —" >&2
    echo "       the shard at $SHARD is un-built/empty or the blacklist ate every task;" >&2
    echo "       refusing to report a zero-task green set as a valid (green) listing." >&2
    exit 1
  fi
  echo "#TOTAL_PRESENT=${#PRESENT[@]}" >&2
  printf '%s\n' "${GREEN[@]}"
  exit 0
fi

# ── 2) run the sweep — content-retry now lives IN run_oracle_sweep.py ──────────
# run_oracle_sweep --all applies black_list.txt (green set = present − blacklist) and
# re-runs its own reward=0 flakes via --content-retries (same _trial_passes gate). This
# wrapper invokes it once and trusts its exit code; set -e aborts on a persistent
# failure (with run_oracle_sweep's own X/N + failed-list summary). The old bash
# re-implementation (_task_ids / _failed_tasks / the retry loop) is gone.
echo "==> run_oracle_sweep --all @ --max-workers $MAX_WORKERS, --content-retries $CONTENT_RETRIES (job-id $JOB_ID)"
echo "    cache=$SHARD  CP=${XRLENV_GRPC_HOST:-<from .env>}"
"$PY" "$SETA/run_oracle_sweep.py" \
  --all --max-workers "$MAX_WORKERS" \
  --content-retries "$CONTENT_RETRIES" \
  --save-artifacts "$JOBS_DIR" --job-id "$JOB_ID" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
echo "======================================================================"
echo "==> ALL GREEN ✅  (see the run_oracle_sweep summary above for the tally)"
