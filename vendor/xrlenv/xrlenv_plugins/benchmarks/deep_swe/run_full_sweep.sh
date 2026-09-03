#!/usr/bin/env bash
# run_full_sweep.sh — one-command DeepSWE oracle sweep (all 113 tasks).
#
# The corpus-quality gate: (re)build the task-dir cache, then run pier's OracleAgent
# per task on the xrlenv cluster and confirm each earns a positive reward. As of
# 2026-07-18 the whole corpus is GREEN (113/113, reward=1.0) — see STATUS.md — so
# the green set is simply "all present tasks" (no EXCLUDE). The EXCLUDE hook + the
# content-retry loop are kept for robustness if a future oracle-sweep surfaces a
# broken/flaky task.
#
# Images are NOT pre-warmed: the cluster's dynamic image cache (lazy-pull-on-acquire
# + LRU eviction + affinity) pulls each task's public-ECR image on first acquire, so
# a full sweep is safe at low concurrency (validated at --max-workers 8).
#
# Typical use:
#     bash xrlenv_plugins/benchmarks/deep_swe/run_full_sweep.sh
#     bash .../run_full_sweep.sh --max-workers 16
#
# Flags (the interface — run knobs are FLAGS, not env vars: a stale exported
# SKIP_BUILD/LIST_GREEN must never silently turn a real sweep into a no-op.
# XRLENV_* deployment config still comes from .env):
#   --max-workers N       trial concurrency (default: 8)
#   --content-retries N   re-run non-passing tasks up to N times (default: 0)
#   --job-id LABEL        run label under tmp/ (default: deepswe-full-sweep; ts appended)
#   --jobs-dir DIR        per-trial artifact root (default: ./tmp)
#   --skip-build-cache    skip the cache (re)build (already built this session)
#   --list-green          print the green set (present − EXCLUDE) and exit; run nothing
#   <anything else>       forwarded to run_oracle_sweep.py
# XRLENV_BENCHMARK_CACHE (cache ROOT) still comes from the env / .env.
#
# TWO independent retry layers (see the README §"Two retry layers — and why"):
#   * --retries 6         per-TRIAL, infra-transient ONLY (CapacityExhausted / node
#                         loss). Set explicitly below; NEVER re-rolls an eval result.
#   * --content-retries 2 per-TASK, re-runs a reward=0 task to tell a one-off
#                         environmental flake from a real content regression (visible: each
#                         retry round writes a sibling <job-id>-retryN/ dir — the summary
#                         reports the final pass/fail outcome, not a retry-count field).
# Timeouts run at each task's NATIVE budget (no --timeout-multiplier) so an
# under-budgeted task fails LOUD rather than being rescued by inflated headroom.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ── flags (the interface) + passthrough for run_oracle_sweep.py ───────────────
LIST_GREEN=0
SKIP_BUILD=0
MAX_WORKERS=8
JOB_ID="deepswe-full-sweep"
CONTENT_RETRIES=0
JOBS_DIR="./tmp"
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
    *)                   PASS_ARGS+=("$1") ;;   # e.g. --timeout-multiplier 1.5
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
SHARD="$XRLENV_BENCHMARK_CACHE/deep-swe"
DS="xrlenv_plugins/benchmarks/deep_swe"
PY=".venv/bin/python"

# Tasks NOT in the green set (empty — the whole corpus is green as of 2026-07-18).
# Excluding by-id (not an include-list) means a re-populate auto-picks up new tasks.
EXCLUDE=()
# ── 1) (re)build the cache ────────────────────────────────────────────────────
if [ "$SKIP_BUILD" = 1 ]; then
  echo "==> --skip-build-cache — skipping build_cache"
else
  echo "==> build_cache --stage all  (populate if missing + patch)"
  "$PY" "$DS/build_cache.py" --stage all
fi

# ── 2) compute the green set = present tasks - EXCLUDE ────────────────────────
if [ ! -d "$SHARD" ]; then
  echo "ERROR: shard not found at $SHARD — did build_cache populate?" >&2
  exit 1
fi
mapfile -t ALL < <(find "$SHARD" -mindepth 1 -maxdepth 1 -type d \
  -exec test -f '{}/task.toml' ';' -printf '%f\n' | sort)
INCLUDE=()
for t in "${ALL[@]}"; do
  skip=0
  for e in "${EXCLUDE[@]:-}"; do [ "$t" = "$e" ] && { skip=1; break; }; done
  [ "$skip" -eq 0 ] && INCLUDE+=("$t")
done
N=${#INCLUDE[@]}

# ── catalog-completeness gate — FATAL, BEFORE the --list-green early return ────
# An incomplete populate must NOT silently define a SMALLER green set that a
# --list-green consumer (xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py) then accepts as "all
# green". A count mismatch is a broken/partial cache, not a warning. Present count
# pinned from STATUS.md; re-pin here if the corpus changes.
if [ "${#ALL[@]}" -ne 113 ]; then
  echo "ERROR: expected 113 present tasks, got ${#ALL[@]} — populate is incomplete;" >&2
  echo "       refusing to define the green set from a partial cache." >&2
  exit 1
fi
# --list-green (xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py sample mode): print the green set
# (present − EXCLUDE) and exit — EXCLUDE stays the single source of known-failing tasks.
if [ "$LIST_GREEN" = 1 ]; then echo "#TOTAL_PRESENT=${#ALL[@]}" >&2; printf '%s\n' "${INCLUDE[@]}"; exit 0; fi
echo "==> present tasks: ${#ALL[@]}  |  excluded: ${#EXCLUDE[@]}  |  green set: $N"
if [ "${#ALL[@]}" -ne 113 ]; then
  echo "WARNING: expected 113 present tasks, got ${#ALL[@]} — populate may be incomplete." >&2
fi
TASKS="$(IFS=,; echo "${INCLUDE[*]}")"

# ── 3) run the sweep (content-retry now lives IN run_oracle_sweep.py) ──────────
# The per-task content-retry (re-run reward=0 flakes up to N times; solved if ANY
# attempt rewards > 0) now lives in run_oracle_sweep.py's --content-retries, using the
# same pass gate. So this wrapper just invokes it once over the green set and trusts
# its exit code (0 iff every task solved after its content-retries). --retries 6 stays
# infra-transient-only (CapacityExhausted / node-loss); an eval result is never
# re-rolled. A --retries in "$@" still overrides (argparse last-wins). Timeouts
# deliberately run at the NATIVE budget — no --timeout-multiplier default.
echo "==> run_oracle_sweep: ${N} task(s) @ --max-workers $MAX_WORKERS, --content-retries $CONTENT_RETRIES (job-id $JOB_ID)"
echo "    cache=$SHARD  CP=${XRLENV_GRPC_HOST:-<from .env>}"
"$PY" "$DS/run_oracle_sweep.py" \
  --tasks "$TASKS" \
  --max-workers "$MAX_WORKERS" \
  --retries 6 \
  --content-retries "$CONTENT_RETRIES" \
  --jobs-dir "$JOBS_DIR" \
  --job-id "$JOB_ID" \
  ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
# set -e aborts here (with run_oracle_sweep's exit code + its own failure summary) if a
# task is still failing after its content-retries; GREEN is reached only on success.
echo "======================================================================"
echo "==> ALL ${N} GREEN ✅"
