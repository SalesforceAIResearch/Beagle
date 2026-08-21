#!/usr/bin/env bash
# Onboard the three reference agents into experiment copies you own, then generate the eval configs.
# Set YOUR_ORG first (the GitHub org/user to create the copies under). Pins upstream release commits;
# --version is the join key generate_eval_configs.py matches against each config's harness.version.
#
# First run creates + seeds the repos. To RE-RUN after the repos already exist (e.g. to apply a new
# --prune, or re-baseline), add --reseed to the relevant command below — it rewrites the `baseline`
# branch (destructive: don't --reseed a repo that already holds candidate branches) and refreshes the
# manifest ref. eval/evolve clone the REMOTE, so the per-container clone follows the seed, not your
# local --dir checkout (delete + re-run to refresh that if you inspect it locally).
set -euo pipefail
set -a; source .env; set +a

: "${YOUR_ORG:?YOUR_ORG must be set before running scripts/onboard_all_agents.sh}"

# monet — <your-org>/monet_code @ develop
python -m beagle.tools.onboard \
    --upstream https://github.com/<your-org>/monet_code --ref 1261608f6530908e3a03218d8f4671b8c7b5b346 \
    --repo "$YOUR_ORG/monet_code_20260816" --private --version 20260816 --branch-name baseline \
    --dir ../beagle-experiments/monet_code_20260816 --profile-name monet_code_20260816
# mini-swe — SWE-agent/mini-swe-agent @ v2.4.6
python -m beagle.tools.onboard \
    --upstream https://github.com/SWE-agent/mini-swe-agent --ref a83fcae82d2a08f0ee0c688f9d137b3566c097f8 \
    --repo "$YOUR_ORG/mini_swe_agent_v2.4.6" --private --version v2.4.6 --branch-name baseline \
    --dir ../beagle-experiments/mini_swe_agent_v2.4.6 --profile-name mini_swe_agent_v2.4.6
# opencode — anomalyco/opencode @ v1.18.16 (release tag). --prune opencode drops the web/desktop/
# marketing apps + demo assets from the seed so every per-container clone is ~12 MB, not ~79 MB
# (patch-safe — see docs/opencode-prune.md).
python -m beagle.tools.onboard \
    --upstream https://github.com/anomalyco/opencode --ref a3647eb025c7615159d417dcc49fc39fdaeba65b \
    --repo "$YOUR_ORG/opencode_v1.18.16" --private --version 1.18.16 --branch-name baseline \
    --prune opencode \
    --dir ../beagle-experiments/opencode_v1.18.16 --profile-name opencode_v1.18.16

# Build runnable configs from the manifests just written (source filled, joined on --version).
python scripts/generate_eval_configs.py
