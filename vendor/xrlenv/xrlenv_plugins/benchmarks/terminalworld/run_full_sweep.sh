#!/usr/bin/env bash
# run_full_sweep.sh — launch the TerminalWorld "green" oracle sweep (187 tasks).
#
# The 187 = all 200 verified tasks MINUS the 13 that can't pass (12 Failed —
# incl. tw_528959's unreasonable timeout — + 1 need-investigation tw_222108,
# per STATUS.md; tw_245032 fixed 2026-07-08). Runs the pinned oracle against the shared
# harbor cache and reports which tasks reward 1.0.
#
# Typical use (after removing the cache dir to force a clean rebuild):
#     rm -rf "$XRLENV_BENCHMARK_CACHE/terminalworld-verified"
#     bash xrlenv_plugins/benchmarks/terminalworld/run_full_sweep.sh
#
# This script:
#   1. (re)builds the cache with `build_cache.py --stage all` — populate (HF
#      download, if the shard is empty) + patch (+cpu-pinning) + sysbox markers.
#      Idempotent: if already populated it just re-applies the overlays (fast).
#   2. computes the green set = present tasks - EXCLUDE (asserts it is 191), and
#   3. runs `run_oracle_sweep.py` over them.
#
# Flags (the interface — run knobs are FLAGS, not env vars: a stale exported
# SKIP_BUILD/LIST_GREEN must never silently turn a real sweep into a no-op.
# XRLENV_* deployment config still comes from .env):
#   --max-workers N       trial concurrency (default: 32)
#   --content-retries N   re-run non-passing tasks up to N times (default: 2)
#   --job-id LABEL        run label under tmp/ (default: tw-full-sweep; ts appended)
#   --jobs-dir DIR        per-trial artifact root (default: ./tmp/sanity-checks)
#   --skip-build-cache    skip step 1 (cache already built this session)
#   --list-green          print the green set (present − EXCLUDE) and exit
#   <anything else>       forwarded to run_oracle_sweep.py (e.g. --timeout-multiplier 1.5 --retries 1)
# XRLENV_BENCHMARK_CACHE (cache ROOT) still comes from the env / .env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ── flags (the interface) + passthrough for run_oracle_sweep.py ───────────────
LIST_GREEN=0
SKIP_BUILD=0
MAX_WORKERS=32
JOB_ID="tw-full-sweep"
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
    *)                   PASS_ARGS+=("$1") ;;   # e.g. --timeout-multiplier 1.5 --retries 1
  esac
  shift
done
# audit Low: --list-green is a READ-ONLY query of the already-built cache — it must NOT
# (re)build (build_cache can need registry creds / mutate the cache). Imply --skip-build-cache.
if [ "$LIST_GREEN" = 1 ]; then SKIP_BUILD=1; fi

# append timestamp to the job id
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
: "${XRLENV_BENCHMARK_CACHE:=/path/to/benchmark-cache}"
export XRLENV_BENCHMARK_CACHE

# audit (cache rename): the old XRLENV_HARBOR_CACHE env var and the old
# .../xrlenv_harbor_cache path are RETIRED. Fail loud so a stale env can't run against the
# wrong (moved/absent) cache and give unreliable results — use XRLENV_BENCHMARK_CACHE +
# /path/to/benchmark-cache. (The Python entrypoints guard this too.)
if [ -n "${XRLENV_HARBOR_CACHE+x}" ]; then
  echo "ERROR: XRLENV_HARBOR_CACHE is retired — unset it and set XRLENV_BENCHMARK_CACHE=/path/to/benchmark-cache" >&2
  exit 1
fi
case "$XRLENV_BENCHMARK_CACHE" in
  *xrlenv_harbor_cache*)
    echo "ERROR: retired cache path '$XRLENV_BENCHMARK_CACHE' — use /path/to/benchmark-cache" >&2
    exit 1 ;;
esac
SHARD="$XRLENV_BENCHMARK_CACHE/terminalworld-verified"
TW="xrlenv_plugins/benchmarks/terminalworld"
PY=".venv/bin/python"

# The 9 tasks NOT in the green set (keep in sync with STATUS.md). Excluding them
# by-id (rather than an include-list) means new green tasks are picked up
# automatically after a re-populate.
#
# 2026-07-17 (step 5, multi-service compose): the 5 compose tasks tw_188260,
# tw_304270, tw_304271, tw_305044, tw_522753 were DROPPED from EXCLUDE — they now
# run through the cluster-compose path (each is a 2-4 service stack on a private
# network). tw_299387 was already green via an in-container sidecar-bootstrap
# workaround, now removed (patches/tw_299387 deleted) so it passes faithfully via
# the real compose sidecars. Green set 187 → 192. Requires the CP + nodes running
# the compose code + the per-task images (incl. tw_188260's solr-node/ambari-server
# sub-builds) pushed — see notes/multi-service-compose-step5-runbook.md.
EXCLUDE=(
  # ❌ Failed — need substrate we don't provide, or a broken/un-completable oracle:
  tw_223822 tw_230695 tw_291556 tw_488034 tw_513637 tw_661946
  # 🔍 Need investigation (deferred): netns-DNS + openal image-version drift:
  tw_222108
  # ⚙️ Unreasonable task config — 2700s ceiling too tight for a CPython
  # from-source `make -j2` build even uncontended (2026-07-08):
  tw_528959
  # 🎲 FLAKY — external-network dependency, NOT retry-recoverable (2026-08-01): the
  # verifier resolves external domains' MX records and asserts on ``nasa.org:``, but
  # nasa.org resolution intermittently fails/times out from the container (redhat.com
  # resolves fine) — ``nasa.org: ERROR`` → reward 0 on every content-retry. It's an
  # external-DNS flake (not xrlenv, not the model), so it's excluded to keep the gate
  # deterministic rather than surprise-failing the sweep. Re-include if the task is
  # rewritten to not depend on live nasa.org DNS.
  tw_507605
)
# ── 1) (re)build the cache ────────────────────────────────────────────────────
if [ "$SKIP_BUILD" = 1 ]; then
  echo "==> --skip-build-cache — skipping build_cache"
else
  echo "==> build_cache --stage all  (populate + patch + cpu-pinning + sysbox)"
  "$PY" "$TW/build_cache.py" --stage all
fi

# ── 2) compute the green set = present tasks - EXCLUDE ────────────────────────
if [ ! -d "$SHARD" ]; then
  echo "ERROR: shard not found at $SHARD — did build_cache populate?" >&2
  exit 1
fi
mapfile -t ALL < <(find "$SHARD" -maxdepth 1 -type d -name 'tw_*' -printf '%f\n' | sort)
INCLUDE=()
for t in "${ALL[@]}"; do
  skip=0
  for e in "${EXCLUDE[@]}"; do [ "$t" = "$e" ] && { skip=1; break; }; done
  [ "$skip" -eq 0 ] && INCLUDE+=("$t")
done
N=${#INCLUDE[@]}

# ── catalog-completeness gate — FATAL, BEFORE the --list-green early return ────
# An incomplete populate or a STATUS.md drift must NOT silently define a SMALLER green
# set that a --list-green consumer (xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py) then accepts as
# "all green". A count mismatch is a broken/partial cache, not a warning. Counts pinned
# from STATUS.md; re-pin here if the corpus changes.
if [ "${#ALL[@]}" -ne 200 ]; then
  echo "ERROR: expected 200 present tasks, got ${#ALL[@]} — populate is incomplete;" >&2
  echo "       refusing to define the green set from a partial cache." >&2
  exit 1
fi
if [ "$N" -ne 191 ]; then
  echo "ERROR: expected 191 green tasks, got $N — shard/STATUS.md drift; refusing to run/list" >&2
  echo "       a green set that doesn't match the pinned catalog." >&2
  exit 1
fi
# --list-green (xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py sample mode): print the green set
# (present − EXCLUDE) and exit — EXCLUDE stays the single source of known-failing tasks.
if [ "$LIST_GREEN" = 1 ]; then echo "#TOTAL_PRESENT=${#ALL[@]}" >&2; printf '%s\n' "${INCLUDE[@]}"; exit 0; fi
echo "==> present tasks: ${#ALL[@]}  |  excluded: ${#EXCLUDE[@]}  |  green set: $N"
if [ "${#ALL[@]}" -ne 200 ]; then
  echo "WARNING: expected 200 present tasks, got ${#ALL[@]} — populate may be incomplete." >&2
fi
if [ "$N" -ne 191 ]; then
  echo "WARNING: expected 191 green tasks, got $N — shard/STATUS.md drift; review before trusting results." >&2
fi
TASKS="$(IFS=,; echo "${INCLUDE[*]}")"

# ── 3) run the sweep (content-retry now lives IN run_oracle_sweep.py) ──────────
# The per-task content-retry (re-run reward=0 flakes — e.g. gdb-backtrace unwind
# tw_234227, DinD-verifier timing tw_650591 — up to N times; solved if ANY attempt
# rewards 1.0) now lives in run_oracle_sweep.py's --content-retries, using the same
# _trial_passes gate. So this wrapper just invokes it once over the green set and
# trusts its exit code (0 iff every task solved after its content-retries). --retries
# stays infra-transient-only. Both drivers — this wrapper AND the ci runner — get the
# same retry from one place; the old bash re-implementation is gone.
echo "==> run_oracle_sweep: ${N} task(s) @ --max-workers $MAX_WORKERS, --content-retries $CONTENT_RETRIES (job-id $JOB_ID)"
echo "    cache=$SHARD  CP=${XRLENV_GRPC_HOST:-<from .env>}"
"$PY" "$TW/run_oracle_sweep.py" \
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
