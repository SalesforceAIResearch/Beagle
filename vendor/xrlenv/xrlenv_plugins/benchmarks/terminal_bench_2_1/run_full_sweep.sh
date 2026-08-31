#!/usr/bin/env bash
# run_full_sweep.sh — launch the terminal-bench-2-1 "green" oracle sweep.
#
# The green set = all present tasks MINUS the EXCLUDE list below (operational
# excludes only — NOT broken oracles). Right now that is 89 present − 1 excluded
# (caffe-cifar-10, whose dataset host is very slow) = 88. Runs the patched oracle
# against the shared harbor cache and reports which tasks reward > 0. An oracle
# FAIL here is a corpus defect (reward ceiling 0 → poison for RL), usually an
# unpinned dependency that drifted — inspect the per-trial verifier output and add
# a pin to build_cache.py's PATCHES table.
#
# Unlike TerminalWorld, tb2.1 tasks carry their OWN ``docker_image`` in task.toml
# (path-source / LocalTaskId), so there is no registry-image-template to export —
# the cluster pulls each task's image on first acquire.
#
# Typical use (after removing the cache dir to force a clean rebuild):
#     rm -rf "$XRLENV_BENCHMARK_CACHE/terminal-bench-2-1"
#     bash xrlenv_plugins/benchmarks/terminal_bench_2_1/run_full_sweep.sh
#
# This script:
#   1. (re)builds the cache with `build_cache.py --stage all` — populate (registry
#      pull, if the shard is empty) + patch (dependency pins on marker-flagged
#      tasks). Idempotent: if already populated it just re-applies the pins (fast).
#   2. computes the green set = present tasks − EXCLUDE (asserts 89 present / 88
#      green), and
#   3. runs `run_oracle_sweep.py` over them.
#
# Flags (the interface — run knobs are FLAGS, not env vars: a stale exported
# SKIP_BUILD/LIST_GREEN must never silently turn a real sweep into a no-op.
# XRLENV_* deployment config still comes from .env):
#   --max-workers N       trial concurrency (default: 32). tb2.1 is all-runc (no
#                         sysbox cap), so this can go high (64 per the README);
#                         xrlenv's fail-fast+retry absorbs any create-time capacity
#                         pressure at high concurrency.
#   --content-retries N   re-run non-passing tasks up to N times (default: 2)
#   --job-id LABEL        run label under tmp/ (default: tb21-full-sweep; ts appended)
#   --jobs-dir DIR        per-trial artifact root (default: ./tmp/sanity-checks)
#   --skip-build-cache    skip step 1 (cache already built this session)
#   --list-green          print the green set (present − EXCLUDE) and exit
#   <anything else>       forwarded to run_oracle_sweep.py (e.g. --timeout-multiplier 1.5 --cpu-pinning)
# XRLENV_BENCHMARK_CACHE (cache ROOT) still comes from the env / .env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ── flags (the interface) + passthrough for run_oracle_sweep.py ───────────────
LIST_GREEN=0
SKIP_BUILD=0
MAX_WORKERS=32
JOB_ID="tb21-full-sweep"
CONTENT_RETRIES=2
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
    *)                   PASS_ARGS+=("$1") ;;   # e.g. --timeout-multiplier 1.5 --cpu-pinning
  esac
  shift
done
# audit Low: --list-green is a READ-ONLY query of the already-built cache — it must NOT
# (re)build (build_cache can need registry creds / mutate the cache). Imply --skip-build-cache.
if [ "$LIST_GREEN" = 1 ]; then SKIP_BUILD=1; fi

# append timestamp to the job id (unique run dir per launch)
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
SHARD="$XRLENV_BENCHMARK_CACHE/terminal-bench-2-1"
TB="xrlenv_plugins/benchmarks/terminal_bench_2_1"
PY=".venv/bin/python"

# Tasks NOT in the green set. Excluding them by-id (rather than an include-list)
# means new tasks are picked up automatically after a re-populate. These are
# OPERATIONAL excludes (infra-friendly), NOT broken oracles — keep the reason
# explicit so a future reader knows it's safe to drop.
EXCLUDE=(
  # 🐢 Operational — the CIFAR-10 dataset download host is very slow, so the
  # oracle routinely busts its wall-clock even though the solve itself is fine.
  # Excluded for now to keep the sweep's signal about OUR infra, not a slow
  # third-party mirror. Revisit if we pre-seed the dataset into the image.
  caffe-cifar-10
)
# ── 1) (re)build the cache ────────────────────────────────────────────────────
if [ "$SKIP_BUILD" = 1 ]; then
  echo "==> --skip-build-cache — skipping build_cache"
else
  echo "==> build_cache --stage all  (populate + patch pinned oracles)"
  "$PY" "$TB/build_cache.py" --stage all
fi

# ── 2) compute the green set = present tasks − EXCLUDE ─────────────────────────
if [ ! -d "$SHARD" ]; then
  echo "ERROR: shard not found at $SHARD — did build_cache populate?" >&2
  exit 1
fi
# A tb2.1 task is a dir with solution/solve.sh (matches the runner's discovery).
mapfile -t ALL < <(
  for d in "$SHARD"/*/; do
    [ -f "${d}solution/solve.sh" ] && basename "$d"
  done | sort
)
INCLUDE=()
for t in "${ALL[@]}"; do
  skip=0
  for e in "${EXCLUDE[@]}"; do [ "$t" = "$e" ] && { skip=1; break; }; done
  [ "$skip" -eq 0 ] && INCLUDE+=("$t")
done
N=${#INCLUDE[@]}

# ── catalog-completeness gate — FATAL, BEFORE the --list-green early return ────
# An incomplete populate or an EXCLUDE drift must NOT silently define a SMALLER green
# set that a --list-green consumer (xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py) then accepts as
# "all green". A count mismatch is a broken/partial cache, not a warning. Counts pinned
# from STATUS.md; re-pin here if the corpus changes.
if [ "${#ALL[@]}" -ne 89 ]; then
  echo "ERROR: expected 89 present tasks, got ${#ALL[@]} — populate is incomplete;" >&2
  echo "       refusing to define the green set from a partial cache." >&2
  exit 1
fi
if [ "$N" -ne 88 ]; then
  echo "ERROR: expected 88 green tasks, got $N — shard/EXCLUDE drift; refusing to run/list" >&2
  echo "       a green set that doesn't match the pinned catalog." >&2
  exit 1
fi
# --list-green (xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py sample mode): print the green set
# (present − EXCLUDE) and exit — EXCLUDE stays the single source of known-failing tasks.
if [ "$LIST_GREEN" = 1 ]; then echo "#TOTAL_PRESENT=${#ALL[@]}" >&2; printf '%s\n' "${INCLUDE[@]}"; exit 0; fi
echo "==> present tasks: ${#ALL[@]}  |  excluded: ${#EXCLUDE[@]} (${EXCLUDE[*]})  |  green set: $N"
echo "==> EXCLUDING for this run: ${EXCLUDE[*]}  (operational — slow dataset host, not a broken oracle)"
if [ "${#ALL[@]}" -ne 89 ]; then
  echo "WARNING: expected 89 present tasks, got ${#ALL[@]} — populate may be incomplete." >&2
fi
if [ "$N" -ne 88 ]; then
  echo "WARNING: expected 88 green tasks, got $N — shard/EXCLUDE drift; review before trusting results." >&2
fi
TASKS="$(IFS=,; echo "${INCLUDE[*]}")"

# ── 3) run the sweep (content-retry now lives IN run_oracle_sweep.py) ──────────
# The per-task content-retry (re-run reward-0 flakes — e.g. portfolio-optimization,
# protein-assembly known flaky under load — up to N times; solved if ANY attempt
# rewards > 0) now lives in run_oracle_sweep.py's --content-retries, using the same
# gate. So this wrapper just invokes it once over the green set and trusts its exit
# code (0 iff every task solved after its content-retries). --retries stays
# infra-transient-only. Both drivers — this wrapper AND the ci runner — get the same
# retry from one place; the old bash re-implementation is gone.
echo "==> run_oracle_sweep: ${N} task(s) @ --max-workers $MAX_WORKERS, --content-retries $CONTENT_RETRIES (job-id $JOB_ID)"
echo "    cache=$SHARD  CP=${XRLENV_GRPC_HOST:-<from .env>}"
"$PY" "$TB/run_oracle_sweep.py" \
  --tasks "$TASKS" \
  --max-workers "$MAX_WORKERS" \
  --content-retries "$CONTENT_RETRIES" \
  --jobs-dir "$JOBS_DIR" \
  --job-id "$JOB_ID" \
  ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
# set -e aborts here (with run_oracle_sweep's exit code + its own failure summary) if a
# task is still failing after its content-retries; GREEN is reached only on success.
echo "======================================================================"
echo "==> ALL ${N} GREEN ✅"
