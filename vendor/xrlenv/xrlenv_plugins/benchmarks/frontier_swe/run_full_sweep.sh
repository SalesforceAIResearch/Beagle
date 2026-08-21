#!/usr/bin/env bash
# run_full_sweep.sh — launch the frontier-swe "green" oracle sweep.
#
# The green set = all ORACLE-GATEABLE present tasks (those shipping
# solution/solve.sh) MINUS the EXCLUDE list below. FrontierSWE is a live
# leaderboard, so upstream withholds the reference solution for 6 corpus tasks —
# those ship NO solution/solve.sh and are therefore never discovered here (they
# cannot be oracle-gated at all). Of the 11 tasks that DO ship a solve.sh, 4 are
# GPU tasks the CPU-only dev cluster can't run, so they are EXCLUDEd — leaving a
# green set of 7. An oracle FAIL here is a corpus/plumbing defect (reward ceiling
# 0 -> poison for RL), not a model signal.
#
# FrontierSWE tasks carry their OWN prebuilt public-GHCR docker_image in task.toml,
# so there is no registry-image-template to export — the cluster pulls each task's
# image on first acquire (or a warm plan pre-pulls it; see build_plan_gen.py).
#
# Typical use (after removing the cache dir to force a clean rebuild):
#     rm -rf "$XRLENV_BENCHMARK_CACHE/frontier-swe"
#     bash xrlenv_plugins/benchmarks/frontier_swe/run_full_sweep.sh
#
# This script:
#   1. (re)builds the cache with `build_cache.py --stage all` (populate git clone,
#      if the shard is empty, + patch). Idempotent.
#   2. computes the green set = present oracle-gateable tasks − EXCLUDE (asserts 11
#      present / 7 green), and
#   3. runs `run_oracle_sweep.py` over them (which owns BOTH retry layers).
#
# Flags (run knobs are FLAGS, not env vars — a stale exported SKIP_BUILD/LIST_GREEN
# must never silently turn a real sweep into a no-op. XRLENV_* deployment config
# still comes from .env):
#   --max-workers N       trial concurrency (default: 8). FrontierSWE tasks are
#                         heavy (up to 128GB/16cpu) and long (oracle ~1-2h), so
#                         the default is conservative; xrlenv's fail-fast+retry
#                         absorbs create-time capacity pressure at higher values.
#   --content-retries N   re-run non-passing tasks up to N times (default: 2)
#   --job-id LABEL        run label under tmp/ (default: frontier-swe-full-sweep; ts appended)
#   --jobs-dir DIR        per-trial artifact root (default: ./tmp/sanity-checks)
#   --skip-build-cache    skip step 1 (cache already built this session)
#   --list-green          print the green set (present − EXCLUDE) and exit
#   <anything else>       forwarded to run_oracle_sweep.py (e.g. --timeout-multiplier 1.5)
# XRLENV_BENCHMARK_CACHE (cache ROOT) still comes from the env / .env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ── flags (the interface) + passthrough for run_oracle_sweep.py ───────────────
LIST_GREEN=0
SKIP_BUILD=0
MAX_WORKERS=8
JOB_ID="frontier-swe-full-sweep"
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
      # The cache root is XRLENV_BENCHMARK_CACHE ONLY. A pass-through --dest reaches
      # the evaluator (argparse last-wins) and would gate one cache but evaluate another.
      echo "ERROR: --dest/--cache is not accepted here — set XRLENV_BENCHMARK_CACHE instead" >&2
      exit 1 ;;
    *)                   PASS_ARGS+=("$1") ;;
  esac
  shift
done
# --list-green is a READ-ONLY query of the already-built cache — it must NOT
# (re)build. Imply --skip-build-cache.
if [ "$LIST_GREEN" = 1 ]; then SKIP_BUILD=1; fi

# append timestamp to the job id (unique run dir per launch)
JOB_ID="${JOB_ID}-$(date +%Y-%m-%d_%H-%M-%S)"

# CP creds (host / token) AND the cache ROOT come from .env. Source it FIRST so the
# ONE resolved XRLENV_BENCHMARK_CACHE below drives EVERY stage. A pure --list-green
# read needs no CP creds (and sourcing .env there would clobber a caller-provided
# fixture cache), so skip .env entirely when listing.
if [ "$LIST_GREEN" != 1 ]; then
  set +u; set -a; source ./.env 2>/dev/null || true; set +a; set -u
fi
: "${XRLENV_BENCHMARK_CACHE:=/path/to/benchmark-cache}"
export XRLENV_BENCHMARK_CACHE

# The old XRLENV_HARBOR_CACHE env var and the old .../xrlenv_harbor_cache path are
# RETIRED. Fail loud so a stale env can't run against the wrong (moved/absent) cache.
if [ -n "${XRLENV_HARBOR_CACHE+x}" ]; then
  echo "ERROR: XRLENV_HARBOR_CACHE is retired — unset it and set XRLENV_BENCHMARK_CACHE=/path/to/benchmark-cache" >&2
  exit 1
fi
case "$XRLENV_BENCHMARK_CACHE" in
  *xrlenv_harbor_cache*)
    echo "ERROR: retired cache path '$XRLENV_BENCHMARK_CACHE' — use /path/to/benchmark-cache" >&2
    exit 1 ;;
esac
SHARD="$XRLENV_BENCHMARK_CACHE/frontier-swe"
FS="xrlenv_plugins/benchmarks/frontier_swe"
PY=".venv/bin/python"

# Tasks NOT in the green set. Excluding them by-id (rather than an include-list)
# means new gateable tasks are picked up automatically after a re-populate.
# 12 tasks ship a solve.sh after the patch stage (11 upstream + notebook-compression's
# xrlenv-authored solution, see below); 5 are excluded here (4 GPU + 1 defect) →
# green set of 7. See STATUS.md for the full evidence. (The other 5 corpus tasks ship
# NO solve.sh, so they are never discovered.)
#
# Two of the 6 green come from curated patches/ overlays, NOT an upstream reference:
#  • dependent-type-checker — a G1 defect FIXED faithfully: its oracle read
#    /tests/reference_impl at solve time; the overlay bundles the byte-identical
#    UPSTREAM reference under solution/ (an oracle, just re-pathed). Confirmed reward 1.0005.
#  • notebook-compression — upstream withholds the reference, so this is an
#    xrlenv-AUTHORED solution (a lossless lzma compressor as the task's /app/run),
#    NOT an upstream oracle. Confirmed on-cluster reward 0.3175 (lossless, 80 files).
# Both are loudly labelled in patches/ and broken out in STATUS.md so the 5 upstream
# oracles are never conflated with the 1 authored solution.
EXCLUDE=(
  # 🖥️  GPU (gpus=1 in task.toml) — the dev cluster is CPU-only, so the oracle can't
  # even schedule. Not a broken oracle; revisit if GPU nodes are added.
  granite-mamba2-inference-optimization
  inference-system-optimization
  optimizer-design
  pcqm4mv2-autoresearch
  # 🧩 UPSTREAM oracle-content defect (G1 evidence in STATUS.md) — NOT an xrlenv/gate
  # issue, and NOT to be papered over by a gate change:
  #  • cranelift: solve.sh is a 7-line PLACEHOLDER ("Oracle solution placeholder —
  #    implement me"). Upstream never wrote the reference; correctness=1.0 is just the
  #    untouched baseline, performance=0. No reachable positive ceiling under its oracle.
  #    (A real reference would have to be authored by upstream — nothing to complete.)
  cranelift-codegen-opt
  # NB: dart-style-haskell WAS excluded here for the xrlenv put_archive 128 MiB limit
  # (its oracle bundles a 340 MB Dart SDK → a 639 MB upload). That limit is fixed
  # (chunked container_put_archive, commit 971602a) and dart is confirmed on-cluster
  # (reward 1.0), so it is now a normal upstream-oracle green — no longer excluded.
)
# ── 1) (re)build the cache ────────────────────────────────────────────────────
if [ "$SKIP_BUILD" = 1 ]; then
  # stderr: --list-green must keep stdout to the green task list ONLY.
  echo "==> --skip-build-cache — skipping build_cache" >&2
else
  echo "==> build_cache --stage all  (git-clone populate + patch)" >&2
  "$PY" "$FS/build_cache.py" --stage all
fi

# ── 2) compute the green set = present oracle-gateable tasks − EXCLUDE ─────────
if [ ! -d "$SHARD" ]; then
  echo "ERROR: shard not found at $SHARD — did build_cache populate?" >&2
  exit 1
fi
# A frontier-swe oracle-gateable task is a dir with solution/solve.sh (matches the
# runner's discovery). Tasks with a withheld reference solution are NOT discovered.
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
# set that a --list-green consumer then accepts as "all green". Counts pinned from
# STATUS.md; re-pin here if the corpus changes.
if [ "${#ALL[@]}" -ne 12 ]; then
  echo "ERROR: expected 12 present gateable tasks (ship solution/solve.sh: 11 upstream +" >&2
  echo "       notebook-compression's authored solution), got ${#ALL[@]} — populate/patch is" >&2
  echo "       incomplete or the corpus drifted; refusing to define the green set." >&2
  exit 1
fi
if [ "$N" -ne 7 ]; then
  echo "ERROR: expected 7 green tasks, got $N — shard/EXCLUDE drift; refusing to run/list" >&2
  echo "       a green set that doesn't match the pinned catalog." >&2
  exit 1
fi
# --list-green (the ci sampler): print the green set (present − EXCLUDE) and exit.
if [ "$LIST_GREEN" = 1 ]; then echo "#TOTAL_PRESENT=${#ALL[@]}" >&2; printf '%s\n' "${INCLUDE[@]}"; exit 0; fi
echo "==> present (gateable): ${#ALL[@]}  |  excluded: ${#EXCLUDE[@]} (4 GPU + 1 defect)  |  green set: $N (${INCLUDE[*]})"
TASKS="$(IFS=,; echo "${INCLUDE[*]}")"

# ── 3) run the sweep (both retry layers live IN run_oracle_sweep.py) ───────────
echo "==> run_oracle_sweep: ${N} task(s) @ --max-workers $MAX_WORKERS, --content-retries $CONTENT_RETRIES (job-id $JOB_ID)"
echo "    cache=$SHARD  CP=${XRLENV_GRPC_HOST:-<from .env>}"
"$PY" "$FS/run_oracle_sweep.py" \
  --tasks "$TASKS" \
  --max-workers "$MAX_WORKERS" \
  --content-retries "$CONTENT_RETRIES" \
  --jobs-dir "$JOBS_DIR" \
  --job-id "$JOB_ID" \
  ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
# set -e aborts here (with run_oracle_sweep's exit code + failure summary) if a task
# is still failing after its content-retries; GREEN is reached only on success.
echo "======================================================================"
echo "==> ALL ${N} GREEN ✅"
