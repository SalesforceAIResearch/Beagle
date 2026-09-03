#!/usr/bin/env bash
# run_full_sweep.sh — the swebench-pro gold-patch (oracle) sweep: the onboarding gate.
#
# Default selection = FULL: all 731 public instances, the corpus as published (before any filtering).
# The other two shipped configurations are one flag away — or use their one-liners:
#   --filtered      the quality-filtered set (scripts/filtered_instance_ids.txt, 478)      -> scripts/run_filtered_sweep.sh
#   --subset-100    the 100-task sample of the filtered set, balanced over the
#                   11 repos (scripts/subset_100_instance_ids.txt)                         -> scripts/run_100_subset_sweep.sh
# Green set = the selection MINUS the EXCLUDE list below (operational excludes with a reason; NOT
# silently-dropped broken oracles — an oracle FAIL is a corpus/plumbing defect to fix in build_cache.py
# or to exclude here explicitly). Images are prebuilt on Docker Hub (jefzda/sweap-images:<tag>) and
# pulled by the cluster on first acquire; warm them first with the configuration's plan (see README).
#
#   1. (re)build the cache: build_cache.py over the selection (idempotent)
#   2. green set = selection − EXCLUDE (catalog-completeness gate: every selected id present)
#   3. run_oracle_sweep.py over the green set (both retry layers live there)
#
# Flags (run knobs are FLAGS, not env vars):
#   --max-workers N       trial concurrency (default: 16; the admission queue gates load, so size it to free cluster cpu/mem — see README)
#   --content-retries N   re-run non-passing tasks up to N times (default: 1)
#   --job-id LABEL        run label (default: swebench-pro-<selection>-sweep); a timestamp is always appended
#   --jobs-dir DIR        per-trial artifact root (default: ./tmp/sanity-checks)
#   --skip-build-cache    skip step 1
#   --list-green          print the green set (selection − EXCLUDE) and exit; read-only (implies --skip-build-cache,
#                         does not source .env — export XRLENV_BENCHMARK_CACHE yourself)
#   --dest / --cache      REJECTED: the cache root is XRLENV_BENCHMARK_CACHE only (audit H9)
#   --filtered / --subset-100   the other two configurations (above)
#   --smoke               the first 8 rows of the dataset (a quick plumbing check; one task: scripts/run_smoke_one.sh)
#   --instances IDS       an explicit comma list of instance ids
#   --ids-file PATH       an explicit id manifest
#   <anything else>       forwarded to run_oracle_sweep.py (e.g. --timeout-multiplier 1.5)
# Env (this repo's .env is sourced):
#   XRLENV_BENCHMARK_CACHE   cache ROOT — tasks live in <root>/swebench-pro/<instance_id>/
#                            (or <root>/swebench-pro/golden_patches/<instance_id>/)  [REQUIRED]
#   The dataset and the upstream kit are FETCHED when unset (both public + ungated), so neither of
#   the two below is required; set one only to pin a local copy, and it is then used verbatim:
#   SWEBENCH_PRO_PARQUET     the dataset parquet, or the directory of a ScaleAI/SWE-bench_Pro snapshot
#   SWEBENCH_PRO_HARNESS     a checkout of https://github.com/scaleapi/SWE-bench_Pro-os
#   XRLENV_GRPC_HOST/_PORT/_TOKEN   the cluster;  XRLENV_PY  interpreter override (default: this repo's .venv)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

LIST_GREEN=0; SKIP_BUILD=0; SMOKE=0; FILTERED=0; SUBSET100=0; IDS_FILE=""; INSTANCES=""; MAX_WORKERS=16; JOB_ID=""; CONTENT_RETRIES=1; JOBS_DIR="./tmp/sanity-checks"
PASS_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --list-green)        LIST_GREEN=1 ;;
    --skip-build-cache)  SKIP_BUILD=1 ;;
    --smoke)             SMOKE=1 ;;
    --filtered)          FILTERED=1 ;;
    --subset-100)        SUBSET100=1 ;;
    --instances)         INSTANCES="$2"; shift ;;
    --ids-file)          IDS_FILE="$2"; shift ;;
    --max-workers)       MAX_WORKERS="$2"; shift ;;
    --content-retries)   CONTENT_RETRIES="$2"; shift ;;
    --job-id)            JOB_ID="$2"; shift ;;
    --jobs-dir)          JOBS_DIR="$2"; shift ;;
    --dest|--dest=*|--cache|--cache=*)
      # H9: the cache root is XRLENV_BENCHMARK_CACHE ONLY. A pass-through --dest would reach
      # build_cache.py / run_oracle_sweep.py and gate one cache while evaluating another.
      echo "ERROR: --dest/--cache is not accepted here — set XRLENV_BENCHMARK_CACHE instead" >&2
      exit 2 ;;
    *)                   PASS_ARGS+=("$1") ;;
  esac
  shift
done
# --list-green is a READ-ONLY query of the already-built cache: never (re)build for it.
if [ "$LIST_GREEN" = 1 ]; then SKIP_BUILD=1; fi

# CP creds AND the cache ROOT come from .env. Source it FIRST so the ONE resolved
# XRLENV_BENCHMARK_CACHE below drives every stage — the shell green-set gate and the Python
# build/eval children (audit H9). A pure --list-green needs no CP creds and must not clobber a
# caller-provided cache root (the integration runner's / a unit test's), so skip .env there.
if [ "$LIST_GREEN" != 1 ]; then
  set +u; set -a; source ./.env 2>/dev/null || true; set +a; set -u
fi
if [ -z "${XRLENV_BENCHMARK_CACHE:-}" ]; then
  echo "ERROR: XRLENV_BENCHMARK_CACHE is not set — export it (cache ROOT; tasks live in <root>/swebench-pro/) or put it in .env" >&2
  exit 2
fi
export XRLENV_BENCHMARK_CACHE
SHARD="$XRLENV_BENCHMARK_CACHE/swebench-pro"
# Same rule as build_cache.shard_dir(): both <root>/swebench-pro/<id>/ and
# <root>/swebench-pro/golden_patches/<id>/ are supported; the canonical one wins when populated.
if ! compgen -G "$SHARD"/*/task.toml >/dev/null 2>&1 && compgen -G "$SHARD/golden_patches"/*/task.toml >/dev/null 2>&1; then
  SHARD="$SHARD/golden_patches"
fi
SP="xrlenv_plugins/benchmarks/swebench_pro"
# interpreter: this repo's venv (uv sync --all-extras: harbor + xrlenv + pyarrow); XRLENV_PY overrides.
PY="${XRLENV_PY:-.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"

# OPERATIONAL excludes only (keep the reason explicit). Broken oracles get fixed or listed here with a reason.
EXCLUDE=(
)

# ── 1) (re)build the cache over the selection (default: full = all 731) ───────
SEL=(--all); LABEL="full"
[ "$FILTERED" = 1 ] && { SEL=(--filtered); LABEL="filtered"; }
[ "$SUBSET100" = 1 ] && { SEL=(--subset-100); LABEL="subset100"; }
[ -n "$IDS_FILE" ] && { SEL=(--ids-file "$IDS_FILE"); LABEL="ids-file"; }
[ -n "$INSTANCES" ] && { SEL=(--instances "$INSTANCES"); LABEL="instances"; }
[ "$SMOKE" = 1 ] && { SEL=(--smoke); LABEL="smoke"; }
[ -z "$JOB_ID" ] && JOB_ID="swebench-pro-${LABEL}-sweep"
JOB_ID="${JOB_ID}-$(date +%Y-%m-%d_%H-%M-%S)"      # every run gets its own artifact dir (harbor refuses to reuse one with a different config)
if [ "$SKIP_BUILD" = 1 ]; then echo "==> --skip-build-cache"; else echo "==> build_cache ${SEL[*]}"; "$PY" "$SP/build_cache.py" "${SEL[@]}"; fi

# ── 2) green set = selection − EXCLUDE, every selected id present in the shard ──
LIST_FILE="$(mktemp)"
"$PY" "$SP/build_cache.py" "${SEL[@]}" --list > "$LIST_FILE" || { echo "ERROR: the selection (${SEL[*]}) did not resolve — see above" >&2; exit 1; }
mapfile -t WANT < "$LIST_FILE"; rm -f "$LIST_FILE"
[ "${#WANT[@]}" -gt 0 ] || { echo "ERROR: the selection (${SEL[*]}) is empty" >&2; exit 1; }
INCLUDE=(); MISSING=()
for t in "${WANT[@]}"; do
  skip=0; for e in "${EXCLUDE[@]}"; do [ "$t" = "$e" ] && { skip=1; break; }; done
  [ "$skip" = 1 ] && continue
  [ -f "$SHARD/$t/solution/solve.sh" ] && INCLUDE+=("$t") || MISSING+=("$t")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "ERROR: ${#MISSING[@]} selected instance(s) not materialized under $SHARD (e.g. ${MISSING[0]}) — populate is incomplete; refusing to define a partial green set" >&2
  exit 1
fi
N=${#INCLUDE[@]}
if [ "$LIST_GREEN" = 1 ]; then echo "#TOTAL_PRESENT=${#WANT[@]}" >&2; printf '%s\n' "${INCLUDE[@]}"; exit 0; fi
echo "==> selection: $LABEL  |  selected: ${#WANT[@]}  |  excluded: ${#EXCLUDE[@]}  |  green set: $N"
TASKS_FILE="$(mktemp)"; printf '%s\n' "${INCLUDE[@]}" > "$TASKS_FILE"

# ── 3) the sweep ──────────────────────────────────────────────────────────────
echo "==> run_oracle_sweep: $N task(s) @ --max-workers $MAX_WORKERS, --content-retries $CONTENT_RETRIES (job-id $JOB_ID)"
"$PY" "$SP/run_oracle_sweep.py" --tasks "$TASKS_FILE" --max-workers "$MAX_WORKERS" --content-retries "$CONTENT_RETRIES" \
  --jobs-dir "$JOBS_DIR" --job-id "$JOB_ID" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
echo "======================================================================"
echo "==> ALL $N GREEN ✅"
