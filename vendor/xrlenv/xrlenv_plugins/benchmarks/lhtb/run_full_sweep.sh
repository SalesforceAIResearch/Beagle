#!/usr/bin/env bash
# run_full_sweep.sh — LHTB oracle sweep (the CI gate).
#
# Runs harbor's OracleAgent per task ON THE XRLENV CLUSTER and gates on the dense
# reward (> 0 = the shipped reference produced a gradable result; LHTB is partial-
# credit). Every one of the 46 tasks is in exactly one of three sets (see STATUS.md):
#
#   GREEN      — expected to reward > 0 under the oracle. Membership depends on MODE:
#                the default §2 path includes the 6 REBUILD tasks (their FIXED images
#                were built + pushed to the private registry in §2); the
#                --use-upstream-image path can't run them (broken on docker.io), so
#                they drop out.
#   TBD        — the 12 issue-#2 tasks that grade a near-exact match to a PRIVATE
#                reference whose schema/constants aren't shipped
#                (github.com/zli12321/LHTB/issues/2). A passing oracle here is a
#                tautology, so they're tracked apart — but still RUN (the oracle does
#                produce a result). Content-validity is TBD.
#   BLACKLIST  — cannot pass even under the oracle until UPSTREAM fixes them. NEVER run.
#                Provisional — finalized once every task issue is triaged.
#
# The sweep RUNS the GREEN + TBD sets and never runs the BLACKLIST. Whatever the mode,
# it prints — LOUDLY — exactly which tasks are excluded and why.
#
# MODE — default validates §2 (the rebuilt private images); opt out for the docker.io gate:
#   (default)              §2 path: build_cache repins the 6 REBUILD tasks at the private
#                          registry (XRLENV_PRIVATE_REGISTRY_HOST, required — usually from
#                          .env); their images must already be built+pushed (README §2).
#                          GREEN includes chess-mate / duckdb / robotics-slam / unknown-config.
#   --use-upstream-image   out-of-box docker.io gate: no repin; the 6 REBUILD tasks are
#                          excluded (their public images are broken/unpublished).
#
# Flags (the interface — not env vars):
#   --use-upstream-image   docker.io out-of-box gate (default: §2 — validate the rebuilds)
#   --max-workers N        trial concurrency (default: 8)
#   --job-id LABEL         run label (default: lhtb-full-sweep; a timestamp is appended)
#   --jobs-dir DIR         where per-trial artifacts land (default: ./tmp)
#   --content-retries N    re-run non-passing tasks up to N times (default: 0)
#   --skip-build-cache     use the cache as-is (skip the build_cache (re)build)
#   --skip-ultra-long      skip the ULTRA_LONG tasks (>30 min oracles) — faster iteration
#   --list-green           print the run set (present − blacklist/rebuild/ultra-long) and exit
#   <anything else>        forwarded to run_oracle_sweep.py (e.g. --timeout-multiplier 2)
# All XRLENV_* config (cache root, private registry, GRPC host/token) comes from .env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ── flags (the interface) + passthrough for run_oracle_sweep.py ───────────────
UPSTREAM=0
SKIP_BUILD=0
SKIP_ULTRA=0
LIST_GREEN=0
MAX_WORKERS=8
JOBS_DIR="./tmp"
JOB_ID="lhtb-full-sweep"
CONTENT_RETRIES=0
PASS_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --use-upstream-image) UPSTREAM=1 ;;
    --skip-build-cache)   SKIP_BUILD=1 ;;
    --skip-ultra-long)    SKIP_ULTRA=1 ;;
    --list-green)         LIST_GREEN=1 ;;
    --max-workers)        MAX_WORKERS="$2"; shift ;;
    --jobs-dir)           JOBS_DIR="$2"; shift ;;
    --job-id)             JOB_ID="$2"; shift ;;
    --content-retries)    CONTENT_RETRIES="$2"; shift ;;
    --dest|--dest=*|--cache|--cache=*)
      # H9: the cache root is XRLENV_BENCHMARK_CACHE ONLY. A pass-through --dest reaches the
      # evaluator (argparse last-wins) and would gate one cache but evaluate another.
      echo "ERROR: --dest/--cache is not accepted here — set XRLENV_BENCHMARK_CACHE instead" >&2
      echo "       (a pass-through cache override would gate one cache and evaluate another, audit H9)." >&2
      exit 1 ;;
    *)                    PASS_ARGS+=("$1") ;;   # e.g. --timeout-multiplier 1.5
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
SHARD="$XRLENV_BENCHMARK_CACHE/lhtb"
DS="xrlenv_plugins/benchmarks/lhtb"
PY=".venv/bin/python"
# ── the three sets (cache dir names; keep in sync with STATUS.md) ─────────────
# BLACKLIST — can't pass even under the oracle until upstream fixes them (provisional):
#   super-mario                 game reference_*.log not shipped; regen needs a torch+net
#                               gen image (deferred, not cheap in build_cache).
#   sudoku-recovery             upstream ref_solver.py needs root to read the root-only
#                               /opt/sudoku/private oracle; harbor runs the oracle as [agent].
#   apex-openroad-ibex-signoff  upstream solve.sh never applies its documented config.mk
#                               fixes (HANDOFF.md defers the closure) -> broken starter config.
BLACKLIST=( super-mario sudoku-recovery apex-openroad-ibex-signoff )

# TBD — issue-#2 grade-against-private-reference (oracle is tautological). RUN, tracked
# apart. gdal-proj-raster-regression is the issue's FAIR counterexample -> GREEN, not here.
TBD=(
  epidemic-inverse-control-audit dicom-radiology-audit matpower-opf-regression
  spice-ephemeris-regression great-expectations-audit
  opensees-seismic-structural-regression-audit nrel-pysam-hybrid-renewables-audit
  epa-swmm-stormwater-regression-audit climate-netcdf-extreme-event-audit
  document-table-layout-reconstruction scientific-figure-data-reconstruction
  materials-phase-diagram-audit
)

# REBUILD — the 6 tasks whose FIXED image we build+push in §2 (single source of truth =
# build_cache.REBUILD_TASKS). GREEN in §2 mode; excluded in --use-upstream-image mode
# (their public images are broken/unpublished). climate/materials are ALSO TBD above.
REBUILD=(
  chess-mate duckdb-optimizer-closure robotics-slam-benchmark-repair
  unknown-config-semantics climate-netcdf-extreme-event-audit materials-phase-diagram-audit
)

# ULTRA_LONG — oracles that run >30 min wall-clock (measured at conc-32). Orthogonal to
# the other sets; dropped ONLY with --skip-ultra-long (opt-in), for faster iteration.
#   unknown-config-semantics  ~88 min (also REBUILD; rate-limited, time-floored 5 stages)
#   nbody-accel-iterative     ~45 min (GREEN; heavy iterative N-body integration)
# Next tier just under 30 min — add here if they cross it under load: riscv-core-debug
# (~29), duckdb-optimizer-closure (~27), vector-db-iterative-build (~22).
ULTRA_LONG=( unknown-config-semantics nbody-accel-iterative )

# ── 1) (re)build the cache — mode-aware; NEVER silently un-repins §2's work ────
if [ "$SKIP_BUILD" = 1 ]; then
  echo "==> --skip-build-cache — using the cache as-is"
elif [ "$UPSTREAM" = "1" ]; then
  echo "==> [out-of-box] build_cache --stage all --use-upstream-image"
  "$PY" "$DS/build_cache.py" --stage all --use-upstream-image
else
  REG="${XRLENV_PRIVATE_REGISTRY_HOST:-}"
  if [ -z "$REG" ]; then
    echo "ERROR: §2 mode needs the private registry — the 6 REBUILD images must be" >&2
    echo "       built + pushed first (README §2). Set XRLENV_PRIVATE_REGISTRY_HOST" >&2
    echo "       (usually via .env), or pass --use-upstream-image for the docker.io gate." >&2
    exit 2
  fi
  echo "==> [§2] build_cache --stage all --registry $REG  (repins the 6 REBUILD tasks;"
  echo "         their images come from the §2 build+push — a missing push 404s at acquire)"
  "$PY" "$DS/build_cache.py" --stage all --registry "$REG"
fi

# ── 2) classify present tasks + compute the run set ───────────────────────────
#   RUN = present − BLACKLIST   (and − REBUILD in --use-upstream-image mode).
#   The run set = GREEN + TBD; only BLACKLIST (and, out-of-box, REBUILD) is dropped.
if [ ! -d "$SHARD" ]; then
  echo "ERROR: shard not found at $SHARD — did build_cache populate?" >&2
  exit 1
fi
mapfile -t ALL < <(find "$SHARD" -mindepth 1 -maxdepth 1 -type d \
  -exec test -f '{}/task.toml' ';' -printf '%f\n' | sort)

declare -A IS_EXCL IS_TBD
for t in "${BLACKLIST[@]}"; do IS_EXCL["$t"]=blacklist; done
if [ "$UPSTREAM" = "1" ]; then
  for t in "${REBUILD[@]}"; do IS_EXCL["$t"]=rebuild; done
fi
if [ "$SKIP_ULTRA" = 1 ]; then
  for t in "${ULTRA_LONG[@]}"; do IS_EXCL["$t"]=ultra-long; done
fi
for t in "${TBD[@]}"; do IS_TBD["$t"]=1; done

RUN=(); GREEN_RUN=(); TBD_RUN=(); EXCL_BLACK=(); EXCL_REBUILD=(); EXCL_ULTRA=()
for t in "${ALL[@]}"; do
  case "${IS_EXCL[$t]:-}" in
    blacklist)  EXCL_BLACK+=("$t"); continue ;;
    rebuild)    EXCL_REBUILD+=("$t"); continue ;;
    ultra-long) EXCL_ULTRA+=("$t"); continue ;;
  esac
  RUN+=("$t")
  if [ "${IS_TBD[$t]:-}" = "1" ]; then TBD_RUN+=("$t"); else GREEN_RUN+=("$t"); fi
done

# ── catalog-completeness gate — FATAL and BEFORE the --list-green early return ──
# An incomplete populate (fewer present tasks than expected) or a BLACKLIST/REBUILD/TBD
# drift must NOT silently define a SMALLER run set that a --list-green consumer (the
# integration gate) then accepts as "all green". A count mismatch here is a broken
# cache / stale set, not a warning. Numbers pinned from STATUS.md §"Gate config":
# present = 46; default §2 run set (GREEN 31 + TBD 12) = 43; --use-upstream-image run
# set (drops the 6 REBUILD tasks → 27 GREEN + 10 TBD) = 37.
LHTB_PRESENT=46
LHTB_RUN_DEFAULT=43       # §2 mode: GREEN + TBD, drops only the 3 BLACKLIST
LHTB_RUN_UPSTREAM=37      # --use-upstream-image: also drops the 6 REBUILD
if [ "${#ALL[@]}" -ne "$LHTB_PRESENT" ]; then
  echo "ERROR: expected $LHTB_PRESENT present tasks, got ${#ALL[@]} — populate is incomplete;" >&2
  echo "       refusing to define the run set from a partial cache." >&2
  exit 1
fi
# The run-set count is checked only when --skip-ultra-long is OFF: that flag legitimately
# drops the ULTRA_LONG tasks, so it defines a smaller-BY-DESIGN set (not a drift).
if [ "$SKIP_ULTRA" = 0 ]; then
  if [ "$UPSTREAM" = "1" ]; then EXPECT_RUN="$LHTB_RUN_UPSTREAM"; else EXPECT_RUN="$LHTB_RUN_DEFAULT"; fi
  if [ "${#RUN[@]}" -ne "$EXPECT_RUN" ]; then
    echo "ERROR: expected $EXPECT_RUN tasks in the run set (mode: $([ "$UPSTREAM" = 1 ] && echo upstream || echo §2)), got ${#RUN[@]} —" >&2
    echo "       shard/BLACKLIST/REBUILD/TBD drift; refusing to run/list a set that doesn't" >&2
    echo "       match the pinned catalog (STATUS.md: $LHTB_PRESENT present / $LHTB_RUN_DEFAULT default / $LHTB_RUN_UPSTREAM upstream)." >&2
    exit 1
  fi
fi

# --list-green (xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py sample mode): print the run set (present −
# blacklist/rebuild/ultra-long — the tasks the gate runs) and exit, single-source.
if [ "$LIST_GREEN" = 1 ]; then echo "#TOTAL_PRESENT=${#ALL[@]}" >&2; printf '%s\n' "${RUN[@]}"; exit 0; fi

# ── SHOUT the mode + the run/exclude breakdown (every run) ────────────────────
if [ "$UPSTREAM" = "1" ]; then
  MODE="--use-upstream-image (docker.io out-of-box)"
else
  MODE="§2 (validate the rebuilt private images)"
fi
echo "======================================================================"
echo "==> MODE: $MODE"
echo "==> present ${#ALL[@]} | running ${#RUN[@]} (GREEN ${#GREEN_RUN[@]} + TBD ${#TBD_RUN[@]}) | excluded $(( ${#EXCL_BLACK[@]} + ${#EXCL_REBUILD[@]} + ${#EXCL_ULTRA[@]} ))"
echo "==> EXCLUDED — blacklist (upstream defect, can't pass even under the oracle):"
echo "        ${EXCL_BLACK[*]:-<none>}"
if [ "${#EXCL_REBUILD[@]}" -gt 0 ]; then
  echo "==> EXCLUDED — rebuild (broken on docker.io; run WITHOUT --use-upstream-image to validate the §2 images):"
  echo "        ${EXCL_REBUILD[*]}"
fi
if [ "${#EXCL_ULTRA[@]}" -gt 0 ]; then
  echo "==> EXCLUDED — ultra-long (--skip-ultra-long; >30 min oracles, drop for faster iteration):"
  echo "        ${EXCL_ULTRA[*]}"
fi
echo "==> RUN·TBD (issue-#2, graded vs a private reference — oracle is tautological, tracked apart):"
echo "        ${TBD_RUN[*]:-<none>}"
echo "======================================================================"

TASKS="$(IFS=,; echo "${RUN[*]}")"
N=${#RUN[@]}

# ── 3) run the sweep (content-retry now lives IN run_oracle_sweep.py) ──────────
# The per-task content-retry (re-run non-passing tasks — reward <= 0 / missing on the
# canonical "reward" key of the dense rewards dict; LHTB is partial-credit, so > 0 = the
# oracle produced a result — up to N times, solved if ANY attempt rewards > 0) now lives
# in run_oracle_sweep.py's --content-retries, using its own _trial_passes gate. So this
# wrapper just invokes it once over the run set and trusts its exit code (0 iff every task
# solved after its content-retries). Both drivers — this wrapper AND the ci runner — get
# the same retry from one place; the old bash re-implementation is gone.
echo "==> run_oracle_sweep: ${N} task(s) @ --max-workers $MAX_WORKERS, --content-retries $CONTENT_RETRIES (job-id $JOB_ID)"
"$PY" "$DS/run_oracle_sweep.py" \
  --tasks "$TASKS" \
  --max-workers "$MAX_WORKERS" \
  --content-retries "$CONTENT_RETRIES" \
  --jobs-dir "$JOBS_DIR" \
  --job-id "$JOB_ID" \
  ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
# set -e aborts here [$MODE] (with run_oracle_sweep's exit code + its own failure
# summary) if a task is still failing after its content-retries; GREEN only on success.
echo "======================================================================"
echo "==> ALL ${N} GREEN ✅  (${#GREEN_RUN[@]} GREEN + ${#TBD_RUN[@]} TBD)"
