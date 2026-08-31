#!/usr/bin/env bash
# EvoClaw — one-command oracle correctness sweep (the 92/98 headline).
#
# EvoClaw is an OUTLIER benchmark (GUIDELINE §7.1, interceptor pattern) that is
# ENTIRELY call-site: even the "build the cache" phase (golden-tar extraction in
# oracle.py) and the "image plan" (image_resolution.py, resolved at run time) run
# inside EvoClaw's own harness. So this entrypoint ships in copy_to_call_site/ and
# runs FROM the EvoClaw checkout — after the payload is staged there.
#
#   # 1) stage this payload into the EvoClaw checkout (README §Setup step 2):
#   rsync -a --delete .../evoclaw/copy_to_call_site/ <EvoClaw-Repo>/xrlenv_onboard/
#   # 2) run the sweep from the checkout:
#   bash <EvoClaw-Repo>/xrlenv_onboard/run_full_sweep.sh
#
# It runs EvoClaw's own batch driver (run_all_xrlenv.py — its name is kept: it is
# EvoClaw's entry point adapted by xrlenv to delegate container/image work to the
# cluster) with the headline correctness config: --apply-yd-fixes + the Table-A cpu
# pins (the 92/98 set). The EvoClaw checkout's .env_private supplies EVOCLAW_DATA_ROOT
# + XRLENV_GRPC_HOST/_PORT + XRLENV_CONSUMER_TOKEN (README §Setup).
#
# Env knobs: WORKERS (default 64), RUN_NAME (default milestone-status-fixes). Extra
# args after the script are forwarded to run_all_xrlenv.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # the staged payload dir (xrlenv_onboard/)
CHECKOUT="$(cd "$HERE/.." && pwd)"                      # the EvoClaw checkout root (parent)
DRIVER="$HERE/run_all_xrlenv.py"
[ -f "$DRIVER" ] || { echo "run_all_xrlenv.py not beside this script ($DRIVER) — stage copy_to_call_site/ first" >&2; exit 2; }

WORKERS="${WORKERS:-64}"
RUN_NAME="${RUN_NAME:-milestone-status-fixes}"

# The headline correctness sweep (92/98): --apply-yd-fixes + the Table-A cpu pins.
# M004/M019 exist in BOTH go-zero and dubbo, so every pin id is repo-qualified. See
# notes/table-a-resource-contention.md in the EvoClaw repo.
PIN=(
  zeromicro_go-zero_v1.6.0_v1.9.3/M004                       # benchmark line-split + p2p contention
  zeromicro_go-zero_v1.6.0_v1.9.3/M019                       # rate-limiter timing test
  apache_dubbo_dubbo-3.3.3_dubbo-3.3.6/M003.1                # thread-pool timing test
  nushell_nushell_0.106.0_0.108.0/milestone_G04_1ddae02      # timing-sensitive under full-98 load
  nushell_nushell_0.106.0_0.108.0/milestone_G04_ca0e961
  navidrome_navidrome_v0.57.0_v0.58.0/milestone_003_sub-01   # persistence DB-race (also n2p eval-retry)
  navidrome_navidrome_v0.57.0_v0.58.0/milestone_003_sub-03
)
PIN_FLAGS=(); for m in "${PIN[@]}"; do PIN_FLAGS+=(--cpu-pinning-milestone "$m"); done

echo ">> running the oracle correctness sweep from $CHECKOUT (workers=$WORKERS, run=$RUN_NAME)" >&2
cd "$CHECKOUT"
exec .venv/bin/python "$DRIVER" \
  --parallelization-level milestone --workers "$WORKERS" \
  --apply-yd-fixes "${PIN_FLAGS[@]}" \
  --run-name "$RUN_NAME" "$@"
