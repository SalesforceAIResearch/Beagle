#!/usr/bin/env bash
# run_full_sweep.sh — launch the swe-rebench "green" oracle sweep.
#
# The green set = all present oracle-gateable tasks (those shipping
# solution/solve.sh — all 860 do) MINUS the EXCLUDE list below. An oracle FAIL
# here is a corpus/plumbing defect (reward ceiling 0 -> poison for RL), not a
# model signal.
#
# SWE-rebench tasks carry their OWN prebuilt Docker Hub docker_image in task.toml
# (written by `build_cache.py --stage repin`), so there is no registry-image
# template to export — the cluster pulls each task's image on first acquire (or a
# warm plan pre-pulls it; see build_plan_gen.py).
#
# Typical use (after removing the cache dir to force a clean rebuild):
#     rm -rf "$XRLENV_BENCHMARK_CACHE/swe-rebench"
#     bash xrlenv_plugins/benchmarks/swe_rebench/run_full_sweep.sh
#
# A full 860-task SWE oracle sweep far exceeds any foreground shell timeout — run
# it under nohup/background and poll.
#
# This script:
#   1. (re)builds the cache with `build_cache.py --stage all` (Harbor Hub populate
#      if the shard is empty, + repin + patch). Idempotent.
#   2. computes the green set = present tasks − EXCLUDE (asserting the pinned
#      catalog count; EXCLUDE is currently empty, so green == present), and
#   3. runs `run_oracle_sweep.py` over them (which owns BOTH retry layers).
#
# Flags (run knobs are FLAGS, not env vars — a stale exported SKIP_BUILD/LIST_GREEN
# must never silently turn a real sweep into a no-op. XRLENV_* deployment config
# still comes from .env):
#   --max-workers N       trial concurrency (default: 32). Callers may request ANY
#                         value; xrlenv paces capacity via fail-fast acquire +
#                         infra-only retries. NEVER lower it to green a red run.
#   --content-retries N   re-run non-passing tasks up to N times (default: 0 —
#                         a zero-tolerance gate. A task that only passes on a
#                         re-run is a finding, not a pass: non-deterministic
#                         reward is noise an RL run inherits. Raise it only to
#                         confirm a suspected flake, never to green a gate.)
#   --job-id LABEL        run label under tmp/ (default: swe-rebench-full-sweep; ts appended)
#   --jobs-dir DIR        per-trial artifact root (default: ./tmp/sanity-checks)
#   --skip-build-cache    skip step 1 (cache already built this session)
#   --list-green          print the green set (present − EXCLUDE) and exit
#   --tasks a,b,c         run only these task ids (intersected with the green set,
#                         so EXCLUDE still applies and a held-out task can't sneak
#                         in). Handled HERE rather than passed through: argparse
#                         last-wins would otherwise let a pass-through --tasks
#                         silently override the green set this script computed.
#   --tasks-file PATH     same, read from a manifest (first token per non-# line
#                         is the id; the rest is a comment) — how scripts/ pins a subset
#   --split YYYY_MM[,..]  run only the upstream MONTHLY split(s). Upstream ships
#                         15 monthly splits whose union is this corpus, but the
#                         Harbor Hub package is flat, so the mapping is restored
#                         from scripts/monthly_splits.json (regenerate it with
#                         scripts/fetch_monthly_splits.py). Composes with --tasks
#                         (the union is taken, then intersected with the green set).
#   --list-splits         print the available splits + task counts, and exit
#   <anything else>       forwarded to run_oracle_sweep.py (e.g. --cpu-pinning)
# XRLENV_BENCHMARK_CACHE (cache ROOT) still comes from the env / .env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Pinned catalog size — the number of tasks `--stage populate` must land. An
# incomplete populate must NOT silently define a SMALLER green set that a
# --list-green consumer then accepts as "all green". Re-pin if the upstream
# dataset grows (SWE-rebench publishes monthly splits; see STATUS.md).
EXPECTED_PRESENT=860

# ── flags (the interface) + passthrough for run_oracle_sweep.py ───────────────
LIST_GREEN=0
SKIP_BUILD=0
MAX_WORKERS=32
JOB_ID="swe-rebench-full-sweep"
CONTENT_RETRIES=0
JOBS_DIR="./tmp/sanity-checks"
WANT_TASKS=""
WANT_SPLITS=""
LIST_SPLITS=0
PASS_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --list-green)        LIST_GREEN=1 ;;
    --skip-build-cache)  SKIP_BUILD=1 ;;
    --max-workers)       MAX_WORKERS="$2"; shift ;;
    --content-retries)   CONTENT_RETRIES="$2"; shift ;;
    --job-id)            JOB_ID="$2"; shift ;;
    --jobs-dir)          JOBS_DIR="$2"; shift ;;
    --tasks)             WANT_TASKS="${WANT_TASKS}${WANT_TASKS:+,}$2"; shift ;;
    --split)             WANT_SPLITS="${WANT_SPLITS}${WANT_SPLITS:+,}$2"; shift ;;
    --list-splits)       LIST_SPLITS=1 ;;
    --tasks-file)
      # Each non-blank, non-# line contributes its FIRST whitespace token as a
      # task id; the rest of the line is a human comment (same convention as
      # seta's black_list.txt), so a manifest can carry per-task notes inline.
      [ -f "$2" ] || { echo "ERROR: --tasks-file $2 not found" >&2; exit 1; }
      WANT_TASKS="${WANT_TASKS}${WANT_TASKS:+,}$(grep -v '^[[:space:]]*#' "$2" \
        | grep -v '^[[:space:]]*$' | awk '{print $1}' | paste -sd,)"; shift ;;
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
if [ -z "${XRLENV_BENCHMARK_CACHE:-}" ]; then
  echo "ERROR: XRLENV_BENCHMARK_CACHE is not set. Set it (in .env or the environment) to your" >&2
  echo "       benchmark cache ROOT — the shared directory build_cache.py populates, e.g." >&2
  echo "       XRLENV_BENCHMARK_CACHE=/path/to/xrlenv_benchmark_cache (see .env.example)." >&2
  exit 1
fi
export XRLENV_BENCHMARK_CACHE

# The old XRLENV_HARBOR_CACHE env var and the old .../xrlenv_harbor_cache path are
# RETIRED. Fail loud so a stale env can't run against the wrong (moved/absent) cache.
if [ -n "${XRLENV_HARBOR_CACHE+x}" ]; then
  echo "ERROR: XRLENV_HARBOR_CACHE is retired — unset it and set XRLENV_BENCHMARK_CACHE=/path/to/xrlenv_benchmark_cache" >&2
  exit 1
fi
case "$XRLENV_BENCHMARK_CACHE" in
  *xrlenv_harbor_cache*)
    echo "ERROR: retired cache path '$XRLENV_BENCHMARK_CACHE' — use /path/to/xrlenv_benchmark_cache" >&2
    exit 1 ;;
esac
SHARD="$XRLENV_BENCHMARK_CACHE/swe-rebench"
FS="xrlenv_plugins/benchmarks/swe_rebench"
PY=".venv/bin/python"
SPLIT_INDEX="$FS/scripts/monthly_splits.json"

# ── monthly splits ────────────────────────────────────────────────────────────
# Upstream publishes monthly splits whose union IS this corpus, but the Harbor
# Hub package we download is the flat `test` split, so nothing in the cache
# records a month. scripts/monthly_splits.json carries the mapping (written by
# scripts/fetch_monthly_splits.py from the HF datasets-server; it is NOT
# derivable from config.json's created_at — see that script's docstring).
# --split resolves to ids here and then flows through the ordinary --tasks
# intersection, so EXCLUDE still applies.
if [ "$LIST_SPLITS" = 1 ] || [ -n "$WANT_SPLITS" ]; then
  [ -f "$SPLIT_INDEX" ] || {
    echo "ERROR: $SPLIT_INDEX not found — regenerate it with:" >&2
    echo "  $PY -m xrlenv_plugins.benchmarks.swe_rebench.scripts.fetch_monthly_splits" >&2
    exit 1; }
fi
if [ "$LIST_SPLITS" = 1 ]; then
  "$PY" -c '
import json, sys
d = json.load(open(sys.argv[1]))["splits"]
for name in sorted(d):
    print(name, len(d[name]), sep="\t")
' "$SPLIT_INDEX"
  exit 0
fi
if [ -n "$WANT_SPLITS" ]; then
  ids="$("$PY" -c '
import json, sys
index, wanted = sys.argv[1], [s.strip() for s in sys.argv[2].split(",") if s.strip()]
d = json.load(open(index))["splits"]
unknown = [s for s in wanted if s not in d]
if unknown:
    sys.stderr.write(
        f"ERROR: unknown split(s) {unknown}. Available: {sorted(d)}\n")
    raise SystemExit(1)
out = sorted({t for s in wanted for t in d[s]})
print(",".join(out))
' "$SPLIT_INDEX" "$WANT_SPLITS")" || exit 1
  WANT_TASKS="${WANT_TASKS}${WANT_TASKS:+,}${ids}"
fi

# ── the EXCLUDE set ───────────────────────────────────────────────────────────
# Excluding by-id (rather than an include-list) means new tasks are picked up
# automatically after a re-populate — SWE-rebench publishes monthly splits.
#
# Currently EMPTY: every one of the 860 tasks is a green-set candidate. Add an id
# here only with MEASURED evidence (an oracle run), and record the reason in
# STATUS.md.
#
# The 2026-09-01 full sweep scored 839/860. Sixteen of those failures were ONE
# infra defect (harbor's nproc-on-a-big-host) and are FIXED — `build_cache.py
# --stage patch` writes their cpu-pinning markers, re-verified 16/16 green.
# Five failures remain, each triaged but deliberately NOT excluded, because
# hiding a red task before it is understood is how a corpus rots:
#
#   CQCL__guppylang-1259          upstream packaging defect — hatchling rejects
#                                 `readme = "../README.md"`, so every F2P and all
#                                 33 P2P come back NOT_FOUND. Needs a patches/
#                                 overlay or an exclusion.
#   canonical__charmcraft-2084    NON-HERMETIC — 4 integration tests hit the
#                                 Charmhub store. Fix is hermeticity, never a retry.
#   sigma67__ytmusicapi-909_interface
#                                 NON-HERMETIC — 14x KeyError: 'auth'; the suite
#                                 wants YouTube Music credentials.
#   modin-project__modin-7434     hangs mid-collection; still hangs WITH pinning
#                                 and at concurrency 1. Unresolved.
#   bluesky__ophyd-async-1165     timing-sensitive: `Failed: Timeout (>1.0s)` on
#                                 test_device_with_children_lazily_connects.
#                                 PASSES solo, fails at conc-32 — the only
#                                 genuinely concurrency-sensitive task in the
#                                 corpus. Do NOT "fix" it by lowering concurrency.
#
# Full evidence (per-task reports, the isolation + pinning experiments): STATUS.md.
EXCLUDE=(
  # NON-HERMETIC — both reach a live third-party API during verification, so no
  # retry, resource change or pinning can make them deterministic. Excluded
  # 2026-09-01 on measured evidence; see STATUS.md.
  #
  #   charmcraft-2084: tests/integration/commands/test_store_commands.py queries
  #   Charmhub for real -> `charmcraft.errors.LibraryError: Library
  #   charms.mysql.v0.mysql not found in Charmhub.` (4 P2P).
  canonical__charmcraft-2084
  #   ytmusicapi-909: hits YouTube Music unauthenticated -> 14x KeyError: 'auth',
  #   plus a response whose shape no longer matches what the task recorded (the
  #   payload carries a live `cver` and `logged_in: 0`). Needs credentials AND
  #   pins nothing about a third-party schema.
  sigma67__ytmusicapi-909_interface

  # UNGATEABLE UPSTREAM CONTENT — the oracle patch is correct; the task cannot
  # be graded deterministically on any hardware. Excluded 2026-09-02.
  #
  #   ophyd-async-1165: its pinned commit 9b567b2 sets a repo-GLOBAL
  #   `[tool.pytest.ini_options] timeout = 1` — one second for every test in the
  #   suite. The P2P `test_device_with_children_lazily_connects` drives a bluesky
  #   RunEngine round-trip through that budget and misses it intermittently
  #   (`Failed: Timeout (>1.0s)` at threading.py:320). Both F2P tests pass every
  #   run; only this P2P flaps. Upstream has since raised the setting to
  #   `timeout = 60`, and the same file already carries two sibling timing tests
  #   they marked xfail("Flaky test") — which XPASS here.
  #   MEASURED, interleaved A/B, 8 pairs: cpus=1 -> 6/8, cpus=8 -> 6/8. Identical,
  #   so this is NOT a resource problem and a bump is not the remedy. cpus=2 and
  #   XRLENV_CPU_PINNING were also tried: no effect. On a loaded box the rate
  #   drops to ~1/4. A task that flips a 25-75% coin is not a gate signal.
  bluesky__ophyd-async-1165
  #   modin-7434: hangs with ZERO progress. It collects 5 items, prints
  #   `test_simple_import`, and emits nothing further — 3 independent runs, with
  #   and without pinning, at concurrency 1 and 32. The 30.4 min runs were the
  #   plug-in's old flat 1800 s exec cap; re-run under its real declared 3000 s
  #   budget it burns the full 52.9 min and raises VerifierTimeoutError. No
  #   partial result to parse, so nothing to grade.
  modin-project__modin-7434
)

# ── 1) (re)build the cache ────────────────────────────────────────────────────
if [ "$SKIP_BUILD" = 1 ]; then
  # stderr: --list-green must keep stdout to the green task list ONLY.
  echo "==> --skip-build-cache — skipping build_cache" >&2
else
  echo "==> build_cache --stage all  (Harbor Hub populate + repin + patch)" >&2
  "$PY" "$FS/build_cache.py" --stage all
fi

# ── 2) compute the green set = present tasks − EXCLUDE ────────────────────────
if [ ! -d "$SHARD" ]; then
  echo "ERROR: shard not found at $SHARD — did build_cache populate?" >&2
  exit 1
fi
# A gateable task is a dir with solution/solve.sh (matches the runner's discovery).
mapfile -t ALL < <(
  for d in "$SHARD"/*/; do
    [ -f "${d}solution/solve.sh" ] && basename "$d"
  done | sort
)

INCLUDE=()
for t in "${ALL[@]}"; do
  skip=0
  for e in ${EXCLUDE[@]+"${EXCLUDE[@]}"}; do [ "$t" = "$e" ] && { skip=1; break; }; done
  [ "$skip" -eq 0 ] && INCLUDE+=("$t")
done
N=${#INCLUDE[@]}

# ── catalog-completeness gate — FATAL, BEFORE the --list-green early return ────
if [ "${#ALL[@]}" -ne "$EXPECTED_PRESENT" ]; then
  echo "ERROR: expected $EXPECTED_PRESENT present gateable tasks (ship solution/solve.sh)," >&2
  echo "       got ${#ALL[@]} — the populate is incomplete or the upstream dataset drifted" >&2
  echo "       (SWE-rebench publishes monthly splits). Refusing to define the green set." >&2
  echo "       If the corpus legitimately grew: re-pin EXPECTED_PRESENT here and" >&2
  echo "       record the new resolved dataset version in STATUS.md." >&2
  exit 1
fi
EXPECTED_GREEN=$(( EXPECTED_PRESENT - ${#EXCLUDE[@]} ))
if [ "$N" -ne "$EXPECTED_GREEN" ]; then
  echo "ERROR: expected $EXPECTED_GREEN green tasks, got $N — shard/EXCLUDE drift;" >&2
  echo "       refusing to run/list a green set that doesn't match the pinned catalog." >&2
  exit 1
fi
# --list-green (the ci sampler): print the green set (present − EXCLUDE) and exit.
# Deliberately BEFORE the --tasks narrowing: --list-green always reports the full
# green set, which is what the sampler in scripts/ consumes.
if [ "$LIST_GREEN" = 1 ]; then echo "#TOTAL_PRESENT=${#ALL[@]}" >&2; printf '%s\n' "${INCLUDE[@]}"; exit 0; fi

# ── 2b) optional --tasks / --tasks-file narrowing ─────────────────────────────
# Intersect with the green set (never a replacement for it), so EXCLUDE still
# applies and a held-out task cannot be smuggled into a run.
GREEN_N=$N
if [ -n "$WANT_TASKS" ]; then
  GREEN_ALL=("${INCLUDE[@]}")
  IFS=',' read -r -a REQUESTED <<< "$WANT_TASKS"
  INCLUDE=(); UNKNOWN=(); HELD_OUT=()
  for t in ${REQUESTED[@]+"${REQUESTED[@]}"}; do
    t="${t//[[:space:]]/}"; [ -n "$t" ] || continue
    in_green=0; present=0
    for g in "${GREEN_ALL[@]}";  do [ "$t" = "$g" ] && { in_green=1; break; }; done
    for a in "${ALL[@]}";        do [ "$t" = "$a" ] && { present=1;  break; }; done
    if [ "$in_green" -eq 1 ]; then INCLUDE+=("$t")
    elif [ "$present" -eq 1 ]; then HELD_OUT+=("$t")
    else UNKNOWN+=("$t"); fi
  done
  if [ "${#UNKNOWN[@]}" -gt 0 ]; then
    echo "ERROR: --tasks names ${#UNKNOWN[@]} task(s) absent from the shard: ${UNKNOWN[*]}" >&2
    echo "       (populate first, or fix the manifest)" >&2
    exit 1
  fi
  if [ "${#HELD_OUT[@]}" -gt 0 ]; then
    echo "==> skipping ${#HELD_OUT[@]} requested task(s) held out by EXCLUDE: ${HELD_OUT[*]}" >&2
    echo "    (see the EXCLUDE list in this script + STATUS.md for why)" >&2
  fi
  if [ "${#INCLUDE[@]}" -eq 0 ]; then
    echo "ERROR: --tasks selected 0 runnable task(s) after intersecting the green set" >&2
    exit 1
  fi
  N=${#INCLUDE[@]}
fi

SEL_NOTE=""
[ -n "$WANT_TASKS" ] && SEL_NOTE="  |  narrowed to: $N"
[ -n "$WANT_SPLITS" ] && SEL_NOTE="$SEL_NOTE  (split $WANT_SPLITS)"
echo "==> present: ${#ALL[@]}  |  excluded: ${#EXCLUDE[@]}  |  green set: $GREEN_N$SEL_NOTE"

# ── 3) run the sweep (both retry layers live IN run_oracle_sweep.py) ───────────
echo "==> run_oracle_sweep: ${N} task(s) @ --max-workers $MAX_WORKERS, --content-retries $CONTENT_RETRIES (job-id $JOB_ID)"
echo "    cache=$SHARD  CP=${XRLENV_GRPC_HOST:-<from .env>}"
TASKS="$(IFS=,; echo "${INCLUDE[*]}")"
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
