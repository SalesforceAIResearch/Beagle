#!/usr/bin/env bash
# restamp_registry_ttl.sh — re-stamp the retention of ALL currently-cached
# content to now + N days, in place.
#
# NORMALLY YOU DON'T NEED THIS. The mirror runs registry:3, which honors
# `proxy.ttl` from config.yml, so new pulls get the configured retention and a
# redeploy is fully config-driven. This script is for the niche case of
# *changing the policy for content already cached*: editing config.yml ttl only
# affects FUTURE pulls, so to push existing entries' expiry out (e.g. bump 90d ->
# 180d immediately, or after migrating from the old registry:2.8.3 which ignored
# the config) you rewrite the scheduler-state expiry directly here. The registry
# is stopped while we edit (it holds state in memory), so it incurs a brief
# (~10s) restart.
#
# Run ON the registry host, as the user who owns the repo/.env (it uses sudo
# only for the root-owned state file):
#   bash scripts/restamp_registry_ttl.sh [days]        # default 90
#
# Cron example (re-extend monthly so the cache never lapses):
#   0 4 1 * *  cd /path/to/xrlenv && bash scripts/restamp_registry_ttl.sh 90 >> /var/log/xrlenv-restamp.log 2>&1
set -euo pipefail

DAYS="${1:-90}"
NAME="${REGISTRY_NAME:-xrlenv-registry-proxy}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${SCRIPT_DIR}/../deploy/registry/run-registry-proxy.sh"
REGISTRY_ENV_FILE="${REGISTRY_ENV_FILE:-${SCRIPT_DIR}/../.env}"

# Resolve the blob-store path from the SAME single source of truth as
# run-registry-proxy.sh: explicit REGISTRY_STORE env > XRLENV_REGISTRY_STORAGE in
# the repo .env > the default. (Reading from .env avoids the operator having to
# remember a second variable when they've set a custom storage path.)
_env_store=""
if [ -f "${REGISTRY_ENV_FILE}" ]; then
    _env_store="$(grep -E '^[[:space:]]*XRLENV_REGISTRY_STORAGE=' "${REGISTRY_ENV_FILE}" 2>/dev/null | tail -1 \
        | cut -d= -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'\$//")" || true
fi
STORE="${REGISTRY_STORE:-${_env_store:-/path/to/data${USER}/xrlenv-registry/proxy}}"
F="${STORE}/scheduler-state.json"

[ -f "$F" ] || { echo "ERROR: no scheduler state at $F (set REGISTRY_STORE or XRLENV_REGISTRY_STORAGE in .env)" >&2; exit 1; }

echo "==> stopping ${NAME} (graceful flush of scheduler state)"
docker stop "${NAME}" >/dev/null

cp "$F" "${F}.bak.$(date +%s)" 2>/dev/null || sudo cp "$F" "${F}.bak.$(date +%s)"

echo "==> re-stamping all cached entries to now + ${DAYS}d"
sudo python3 - "$F" "$DAYS" <<'PY'
import json, sys, datetime
f, days = sys.argv[1], int(sys.argv[2])
d = json.load(open(f))
exp = (datetime.datetime.utcnow() + datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
n = 0
for v in d.values():
    if isinstance(v, dict) and "ExpiryData" in v:
        v["ExpiryData"] = exp
        n += 1
json.dump(d, open(f, "w"))
print(f"  re-stamped {n} entries -> {exp}")
PY

echo "==> restarting ${NAME}"
bash "${RUN}" 2>&1 | grep -E 'ttl|/v2/|started' || true
echo "==> done. Cache retained until ~$(date -u -d "+${DAYS} days" +%Y-%m-%d 2>/dev/null || echo "+${DAYS}d")."
