#!/usr/bin/env bash
# restamp_registry_ttl.sh — re-stamp the retention of ALL currently-cached
# content to now + N days, in place.
#
# NORMALLY YOU DON'T NEED THIS. The mirror runs registry:3, which honors
# `proxy.ttl` from config-mirror.yml, so new pulls get the configured retention and a
# redeploy is fully config-driven. This script is for the niche case of
# *changing the policy for content already cached*: editing config-mirror.yml ttl only
# affects FUTURE pulls, so to push existing entries' expiry out (e.g. bump 90d ->
# 180d immediately, or after migrating from the old registry:2.8.3 which ignored
# the config) you rewrite the scheduler-state expiry directly here. The registry
# is stopped while we edit (it holds state in memory), so it incurs a brief
# (~10s) restart.
#
# Run ON the registry host, as the user who owns the repo/.env (it uses sudo
# only for the root-owned state file):
#   bash deploy/registry/restamp_registry_ttl.sh [days]        # default 90
#
# Cron example (re-extend monthly so the cache never lapses):
#   0 4 1 * *  cd /path/to/xrlenv && bash deploy/registry/restamp_registry_ttl.sh 90 >> /var/log/xrlenv-restamp.log 2>&1
set -euo pipefail

DAYS="${1:-90}"
NAME="${REGISTRY_NAME:-xrlenv-registry-proxy}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${SCRIPT_DIR}/run-registry-mirror.sh"
REGISTRY_ENV_FILE="${REGISTRY_ENV_FILE:-${SCRIPT_DIR}/../../.env}"

# Resolve the blob-store path from the SAME single source of truth as
# run-registry-mirror.sh: explicit REGISTRY_STORE env > XRLENV_MIRROR_REGISTRY_STORAGE
# in the repo .env (falling back to the deprecated XRLENV_REGISTRY_STORAGE with a
# warning) > the default.
_read_env_key() {  # $1 = key; echoes the .env value or empty
    [ -f "${REGISTRY_ENV_FILE}" ] || return 0
    grep -E "^[[:space:]]*$1=" "${REGISTRY_ENV_FILE}" 2>/dev/null | tail -1 \
        | cut -d= -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
              -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'\$//" || true
}
_env_store="$(_read_env_key XRLENV_MIRROR_REGISTRY_STORAGE)"
if [ -z "${_env_store}" ]; then
    _env_store="$(_read_env_key XRLENV_REGISTRY_STORAGE)"
    [ -n "${_env_store}" ] && echo "WARN: XRLENV_REGISTRY_STORAGE is deprecated — rename to XRLENV_MIRROR_REGISTRY_STORAGE." >&2
fi
STORE="${REGISTRY_STORE:-${_env_store:-/fsx/home/${USER}/xrlenv-registry/proxy}}"
F="${STORE}/scheduler-state.json"

[ -f "$F" ] || { echo "ERROR: no scheduler state at $F (set REGISTRY_STORE or XRLENV_MIRROR_REGISTRY_STORAGE in .env)" >&2; exit 1; }

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
