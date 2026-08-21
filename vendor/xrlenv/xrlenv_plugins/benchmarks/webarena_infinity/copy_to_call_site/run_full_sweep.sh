#!/usr/bin/env bash
# WebArena-Infinity — the full real-tasks sweep, run FROM the WAI checkout.
#
# WAI is an OUTLIER benchmark (GUIDELINE §7.2, runner-shim pattern): the eval runs
# from inside the WAI checkout (the "call site"), because the integration scripts
# import WAI's own evaluation/ modules and drive an in-container agent+verifier loop.
# So this entrypoint ships in copy_to_call_site/ and runs FROM the WAI checkout —
# after the two host-side prep steps are done (README §"Run an eval, end to end"):
#
#   # 1) (host, once per image change) build + push the answer-free substrate:
#   scripts/build_and_push_images.py --plan .../webarena_infinity/build_plan.yaml --force
#   # 2) stage this payload into the WAI checkout's evaluation/:
#   cp .../webarena_infinity/copy_to_call_site/* <WAI-Repo>/evaluation/
#   # 3) run the FULL real-tasks sweep from the checkout:
#   bash <WAI-Repo>/evaluation/run_full_sweep.sh
#
# The build+push (step 1) is the separate image-prep phase (host-side; the Dockerfile
# + build_plan.yaml live at this benchmark's top level) — deliberately NOT in here, the
# same way golden-path `xrlenv build apply` is separate from run_full_sweep.sh.
#
# DEFAULT = the OFFICIAL 10-app real-tasks set (the paper figure "Performance of
# Browser-use Agents"; ~1260 real-tasks) — the exact apps in coding-bench
# configs/test_wai_monet_full_real-tasks.yaml. MODEL defaults to `oracle` — the
# browserless answer-injection gate (no LLM/creds) that validates the pipeline + every
# task's verifier end-to-end. Set APPS=all for the full 13-app HF corpus (1620 tasks).
#
# SCHEDULING: the underlying runner (run_eval_parallel_xrlenv.py) takes ONE --web-app, so
# this loops the apps **sequentially** (parallel WITHIN an app via --workers; workers idle
# at app boundaries). This differs from coding-bench's real monet eval, which uses a single
# **global app-ordered queue** (parallelism N across ALL tasks) via `runner.run` — use that
# config for the efficient leaderboard run. Sequential is fine for the oracle gate.
#
# Env knobs:
#   MODEL      default oracle        # or gpt / gemini-pro / … (a real agent needs LLM keys in .env)
#   TASK_SUITE default real-tasks    # or function-tasks / all
#   WORKERS    default 8             # containers in flight cluster-wide (per app)
#   APPS       unset                 # "all" = every app with the suite (13); or an explicit
#                                    #   space-separated list, e.g. "apps/gmail apps/xero-invoicing"
#   WEB_APP    unset                 # restrict to ONE app, e.g. apps/gmail
# Extra args are forwarded to run_eval_parallel_xrlenv.py. The WAI repo .env supplies
# XRLENV_GRPC_HOST + XRLENV_CONSUMER_TOKEN + XRLENV_PRIVATE_REGISTRY_HOST/_PORT + LLM keys.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # the staged payload dir (evaluation/)
CHECKOUT="$(cd "$HERE/.." && pwd)"                      # the WAI checkout root (parent of evaluation/)
DRIVER="$HERE/run_eval_parallel_xrlenv.py"
[ -f "$DRIVER" ] || { echo "run_eval_parallel_xrlenv.py not beside this script ($DRIVER) — stage copy_to_call_site/ first" >&2; exit 2; }

MODEL="${MODEL:-oracle}"
TASK_SUITE="${TASK_SUITE:-real-tasks}"
WORKERS="${WORKERS:-8}"

# The OFFICIAL 10 (paper figure) — coding-bench configs/test_wai_monet_full_real-tasks.yaml.
OFFICIAL_APPS=(
  apps/elation-clinical-records   apps/elation-prescriptions       apps/gitlab-plan-and-track
  apps/gmail                      apps/gmail-accounts-and-contacts apps/handshake-career-exploration
  apps/linear-account-settings    apps/paypal-my-wallet            apps/superhuman-general
  apps/xero-invoicing
)

cd "$CHECKOUT"

# Resolve the app list: single WEB_APP > explicit/`all` APPS > the OFFICIAL 10.
if [ -n "${WEB_APP:-}" ]; then
  app_list=("$WEB_APP")
elif [ "${APPS:-}" = "all" ]; then
  app_list=(); for d in apps/*/; do [ -d "${d}${TASK_SUITE}" ] && app_list+=("apps/$(basename "$d")"); done
elif [ -n "${APPS:-}" ]; then
  read -r -a app_list <<< "$APPS"
else
  app_list=("${OFFICIAL_APPS[@]}")
fi

# Keep only apps that actually ship the requested suite in this checkout (warn + skip the
# rest, so a dataset-name drift degrades gracefully instead of failing the whole sweep).
resolved=()
for app in "${app_list[@]}"; do
  if [ -d "$CHECKOUT/$app/$TASK_SUITE" ]; then resolved+=("$app")
  else echo "WARN: skipping $app — no $TASK_SUITE/ under $CHECKOUT/$app" >&2; fi
done
[ "${#resolved[@]}" -gt 0 ] || { echo "no apps with a ${TASK_SUITE}/ suite to run — is the WAI checkout complete?" >&2; exit 2; }

echo ">> WAI sweep from $CHECKOUT: ${#resolved[@]} app(s), suite=$TASK_SUITE, model=$MODEL, workers=$WORKERS (sequential per app)" >&2
rc=0
for app in "${resolved[@]}"; do
  echo ">> === $app ($TASK_SUITE) ===" >&2
  .venv/bin/python evaluation/run_eval_parallel_xrlenv.py \
    --model "$MODEL" --web-app "$app" --task-suite "$TASK_SUITE" --workers "$WORKERS" "$@" || rc=1
done
exit "$rc"
