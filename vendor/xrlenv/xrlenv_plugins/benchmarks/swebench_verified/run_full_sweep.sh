#!/usr/bin/env bash
# run_full_sweep.sh — SWE-bench Verified oracle sweep (the CI gate).
#
# The corpus-quality gate: (re)build the task-data cache, then drive upstream's
# swebench harness (docker-py drop-in) per instance on the xrlenv cluster with the
# GOLD patch as the prediction, and confirm each instance's report.json shows
# ``resolved: true``. Green set = present cached instances − EXCLUDE (empty today —
# every gold-patch-as-prediction should resolve; see STATUS.md). An oracle FAIL is a
# plumbing/content defect (a resolvable instance that didn't), not a model signal.
#
# swebench images are prebuilt on Docker Hub (swebench/sweb.eval.x86_64.*): the
# cluster's dynamic image cache pulls each on first acquire (no pre-warm required;
# `xrlenv build apply` the plan for a big sweep to amortize the pull).
#
# Typical use:
#     bash xrlenv_plugins/benchmarks/swebench_verified/run_full_sweep.sh
#     bash .../run_full_sweep.sh --max-workers 8
#
# Flags (the interface — run knobs are FLAGS, not env vars: a stale exported
# SKIP_BUILD/LIST_GREEN must never silently turn a real sweep into a no-op.
# XRLENV_* deployment config still comes from .env):
#   --max-workers N       trial concurrency (default: 8; swebench's harness is thread-safe)
#   --content-retries N   re-run non-resolved instances up to N times (default: 0)
#   --job-id LABEL        run label under tmp/ (default: swebench-verified-full-sweep; ts appended)
#   --jobs-dir DIR        per-run artifact root (default: ./tmp)
#   --skip-build-cache    skip the cache (re)build (already built this session)
#   --list-green          print the green set (present − EXCLUDE) and exit; run nothing
#   <anything else>       forwarded to run_oracle_sweep.py (e.g. --local, --timeout 3600)
# XRLENV_BENCHMARK_CACHE (cache ROOT) still comes from the env / .env.
#
# TWO retry layers (same design as the harbor/pier sweeps):
#   * --retries 6         per-TRIAL, infra-transient ONLY (CapacityExhausted / node loss);
#                         set explicitly below; NEVER re-rolls a resolution outcome.
#   * --content-retries 2 per-INSTANCE, re-runs a non-resolved instance to tell a one-off
#                         environmental flake from a real regression (counts reported).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ── flags (the interface) + passthrough for run_oracle_sweep.py ───────────────
LIST_GREEN=0
SKIP_BUILD=0
SMOKE=0
MAX_WORKERS=8
JOB_ID="swebench-verified-full-sweep"
CONTENT_RETRIES=0
JOBS_DIR="./tmp"
PASS_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --list-green)        LIST_GREEN=1 ;;
    --skip-build-cache)  SKIP_BUILD=1 ;;
    --smoke)             SMOKE=1 ;;   # intentional 8-instance subset (not the full suite)
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
    *)                   PASS_ARGS+=("$1") ;;   # e.g. --local, --timeout 3600
  esac
  shift
done
# audit Low: --list-green is a READ-ONLY query of the already-built cache — it must NOT
# (re)build (build_cache can need registry creds / mutate the cache). Imply --skip-build-cache.
if [ "$LIST_GREEN" = 1 ]; then SKIP_BUILD=1; fi


# SWE-bench Verified is exactly 500 instances. The FULL suite requires all 500 present;
# a smoke/partial cache must NOT masquerade as the suite and green a subset (audit H4).
SWEBENCH_VERIFIED_TOTAL=500
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
SHARD="$XRLENV_BENCHMARK_CACHE/swebench-verified"
SB="xrlenv_plugins/benchmarks/swebench_verified"
PY=".venv/bin/python"

# Instances NOT in the green set. Excluding by-id (not an include-list) means a
# re-populate auto-picks up new instances. These are UPSTREAM-ungradeable under
# swebench 4.1.0 + the canonical Docker Hub images — the gold-patch oracle itself
# cannot resolve them (NOT an xrlenv/cache/mirror defect). Full root cause + how to
# re-include each is in STATUS.md. The H4 membership gate below is reconciled so
# INCLUDE ∪ EXCLUDE must STILL equal the 500-id manifest: a held-out id is a real
# Verified instance, not a fabrication or a silent subset.
EXCLUDE=(
  # UPSTREAM-ungradeable: the gold-patch oracle itself cannot resolve under swebench 4.1.0 +
  # the canonical Docker Hub images (fail even in isolation). Root cause per id in STATUS.md.
  #
  # NOTE: the 4 flaky psf__requests (non-hermetic external httpbin.org) are intentionally NOT
  # excluded — they're kept IN the green set and documented as flaky in STATUS.md. A one-off
  # 503 under concurrency is therefore an expected, no-surprise flake; a CONSISTENT failure is
  # a real new signal worth investigating (rather than hidden by exclusion).
  sphinx-doc__sphinx-8595    # upstream eval.sh: new-files-only test_patch -> bare `git checkout <base>` reverts pre_install's `pytest -rA` -> parser gets 0 markers on a PASSING test
  sphinx-doc__sphinx-9711    # same new-files-only bare-checkout `-rA` revert
  astropy__astropy-8872      # image setuptools-68 distutils DeprecationWarning + astropy filterwarnings=error -> collection error (0/80)
  astropy__astropy-8707      # same setuptools-68 distutils DeprecationWarning -> 7 tests error
  astropy__astropy-7606      # dataset expects test_compose_roundtrip[] but current astropy collects [unit0]/[%]/... -> expected id never appears
  django__django-10097       # GenericInlineAdminWithUniqueTogetherTest test_add/test_delete ERROR deterministically in the canonical env
)
# ── 1) (re)build the cache ────────────────────────────────────────────────────
if [ "$SKIP_BUILD" = 1 ]; then
  echo "==> --skip-build-cache — skipping build_cache"
else
  echo "==> build_cache --stage all  (materialize Verified rows into the cache shard)"
  "$PY" "$SB/build_cache.py" --stage all --all
fi

# ── 2) compute the green set = present cached instances − EXCLUDE ──────────────
if [ ! -d "$SHARD" ]; then
  echo "ERROR: shard not found at $SHARD — did build_cache populate?" >&2
  exit 1
fi
# Enumerate COMPLETE instance dirs via build_cache's own semantic check (audit M13) — NOT a
# mere "instance.json exists" glob. A dir with a bare/corrupt anchor (no matching extracts,
# an id that disagrees with the dir name, missing required fields) is NOT a prepared instance
# and must not be counted toward the corpus; otherwise a 500-dir cache of ``{}`` anchors would
# pass membership and green the gate, then fail later while loading each row.
mapfile -t ALL < <("$PY" "$SB/build_cache.py" --list-complete | sort)
INCLUDE=()
for t in "${ALL[@]}"; do
  skip=0
  for e in "${EXCLUDE[@]:-}"; do [ "$t" = "$e" ] && { skip=1; break; }; done
  [ "$skip" -eq 0 ] && INCLUDE+=("$t")
done
N=${#INCLUDE[@]}

# ── catalog-completeness gate — FATAL, BEFORE the --list-green early return ────
if [ "$SMOKE" = 1 ]; then
  # Intentional 8-instance subset. Require the smoke ids actually present (single source =
  # build_cache.SMOKE_INSTANCES), so --smoke can't silently green a smaller-than-8 cache.
  mapfile -t SMOKE_IDS < <("$PY" -c "from xrlenv_plugins.benchmarks.swebench_verified.build_cache import SMOKE_INSTANCES; print('\n'.join(SMOKE_INSTANCES))")
  GREEN=()
  for s in "${SMOKE_IDS[@]}"; do
    hit=0; for t in "${INCLUDE[@]}"; do [ "$t" = "$s" ] && { hit=1; break; }; done
    if [ "$hit" -eq 1 ]; then GREEN+=("$s"); else
      echo "ERROR: --smoke needs the ${#SMOKE_IDS[@]} smoke instances; missing $s from the cache." >&2
      exit 1
    fi
  done
  INCLUDE=("${GREEN[@]}"); N=${#INCLUDE[@]}
else
  # FULL suite: the cache must BE the authoritative 500-instance Verified corpus BY
  # MEMBERSHIP against the vendored manifest (audit H4) — not just count, so 500 fabricated
  # ids can't pass. Build all 500 with build_cache.py --all, or pass --smoke for the subset.
  #
  # Read the manifest through the VALIDATED reader (audit M11), NOT a raw ``grep`` over the
  # file: read_verified_manifest enforces count==500, uniqueness, sort order, and a MANDATORY
  # sha256 digest — so a corrupt 499-line manifest (plus a matching 499-id cache) can't
  # silently become the authority. A validation failure fails the gate CLOSED.
  if ! MANIFEST_IDS="$("$PY" -c '
import sys
from xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen import read_verified_manifest
ids = read_verified_manifest()
if ids is None:
    sys.stderr.write("manifest failed validation\n"); sys.exit(3)
sys.stdout.write("\n".join(ids))
')"; then
    echo "ERROR: authoritative Verified manifest is missing/invalid" >&2
    echo "       (count/uniqueness/order/digest) — refusing to gate against an untrusted" >&2
    echo "       corpus definition. Regenerate verified_instance_ids.txt." >&2
    exit 1
  fi
  # INCLUDE (the RUN set) plus EXCLUDE (documented held-out, see STATUS.md) must TOGETHER
  # reconstruct the manifest EXACTLY — a held-out id stays accounted for as a real Verified
  # instance (audit H4: no fabrication, no silent subset). Three fail-closed checks:
  #   missing     = manifest ids neither run NOR excluded (a genuinely absent instance)
  #   extra       = run ids not in the manifest (fabricated)
  #   bad_exclude = EXCLUDE ids not in the manifest (a typo'd holdout that would silently no-op)
  present_or_excluded=$(printf '%s\n' "${INCLUDE[@]}" ${EXCLUDE[@]+"${EXCLUDE[@]}"} | sort -u)
  missing=$(comm -23 <(printf '%s\n' "$MANIFEST_IDS" | sort -u) \
                     <(printf '%s\n' "$present_or_excluded"))
  extra=$(comm -13 <(printf '%s\n' "$MANIFEST_IDS" | sort -u) \
                   <(printf '%s\n' "${INCLUDE[@]}" | sort -u))
  bad_exclude=$(comm -13 <(printf '%s\n' "$MANIFEST_IDS" | sort -u) \
                         <(printf '%s\n' ${EXCLUDE[@]+"${EXCLUDE[@]}"} | sort -u))
  if [ -n "$missing" ] || [ -n "$extra" ] || [ -n "$bad_exclude" ]; then
    nmiss=$(printf '%s\n' "$missing" | grep -c . || true)
    nextra=$(printf '%s\n' "$extra" | grep -c . || true)
    nbadx=$(printf '%s\n' "$bad_exclude" | grep -c . || true)
    echo "ERROR: cache ∪ EXCLUDE is NOT the authoritative Verified corpus (audit H4): ${nmiss} missing," >&2
    echo "       ${nextra} unexpected, ${nbadx} bad-exclude id(s) vs the ${SWEBENCH_VERIFIED_TOTAL}-id manifest." >&2
    echo "       Run 'build_cache.py --stage all --all', or pass --smoke for the 8-id subset." >&2
    [ -n "$extra" ]       && { echo "  unexpected (first 5):"  >&2; printf '%s\n' "$extra"       | head -5 | sed 's/^/    /' >&2; }
    [ -n "$missing" ]     && { echo "  missing (first 5):"     >&2; printf '%s\n' "$missing"     | head -5 | sed 's/^/    /' >&2; }
    [ -n "$bad_exclude" ] && { echo "  bad EXCLUDE (first 5):" >&2; printf '%s\n' "$bad_exclude" | head -5 | sed 's/^/    /' >&2; }
    exit 1
  fi
fi

# --list-green (xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py sample mode): print the green set
# (present − EXCLUDE, or the smoke subset) and exit — EXCLUDE stays the single source.
if [ "$LIST_GREEN" = 1 ]; then echo "#TOTAL_PRESENT=${#ALL[@]}" >&2; printf '%s\n' "${INCLUDE[@]}"; exit 0; fi
echo "==> present instances: ${#ALL[@]}  |  excluded: ${#EXCLUDE[@]}  |  green set: $N"
if [ "$N" -eq 0 ]; then
  echo "ERROR: green set is empty — is the cache populated?" >&2
  exit 1
fi

# ── 3) run the sweep — content-retry now lives IN run_oracle_sweep.py ──────────
# run_oracle_sweep re-runs its own UNRESOLVED instances via --content-retries (an
# instance is solved if ANY attempt resolves — upstream's own `resolved` field), so
# this wrapper invokes it once over the green set and trusts its exit code. --retries 6
# stays infra-transient-only (never re-rolls a resolution). set -e aborts on a
# persistent failure (with run_oracle_sweep's own X/N + failed-list summary).
TASKS="$(IFS=,; echo "${INCLUDE[*]}")"
echo "==> run_oracle_sweep: ${N} instance(s) @ --max-workers $MAX_WORKERS, --content-retries $CONTENT_RETRIES (job-id $JOB_ID)"
echo "    cache=$SHARD  CP=${XRLENV_GRPC_HOST:-<from .env>}"
"$PY" "$SB/run_oracle_sweep.py" \
  --tasks "$TASKS" \
  --max-workers "$MAX_WORKERS" \
  --retries 6 \
  --content-retries "$CONTENT_RETRIES" \
  --jobs-dir "$JOBS_DIR" \
  --job-id "$JOB_ID" \
  ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
echo "======================================================================"
echo "==> ALL ${N} GREEN ✅"
