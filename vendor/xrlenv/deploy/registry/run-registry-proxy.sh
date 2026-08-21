#!/usr/bin/env bash
# run-registry-proxy.sh — start/restart the xrlenv pull-through registry mirror
# on the control-plane box. Reads its config from a single .env file (your
# source of truth — the repo-root .env by default), so you DON'T hand-maintain a
# separate secrets file.
#
# Keys it reads from the env file (all optional; calling-env values win):
#   DOCKERHUB_USER / DOCKERHUB_TOKEN   upstream Docker Hub auth — mapped to the
#                                      registry's REGISTRY_PROXY_USERNAME/PASSWORD
#                                      so the proxy isn't anonymous (the anon cap
#                                      is ~100 pulls/6h; a Pro/Team PAT is much
#                                      higher). Same two keys bootstrap/refresh
#                                      already use for node-side docker auth.
#   XRLENV_REGISTRY_STORAGE            FSx blob-store path
#                                      (default /path/to/data$USER/xrlenv-registry/proxy;
#                                      e.g. set /path/to/data/xrlenv-registry)
#   XRLENV_REGISTRY_PORT               host port (default 5010)
#
# Override the env-file path with REGISTRY_ENV_FILE=/path/to/.env.
#
# Usage (on the control-plane box):
#   bash deploy/registry/run-registry-proxy.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REGISTRY_ENV_FILE="${REGISTRY_ENV_FILE:-${REPO_ROOT}/.env}"

# Read KEY=value from the env file WITHOUT executing it (the .env is
# hand-maintained, so parse rather than `source` to avoid surprises). A value
# already set in the calling environment wins. Strips surrounding quotes.
read_env() {
    local key="$1" val=""
    if [ -n "${!key:-}" ]; then printf '%s' "${!key}"; return 0; fi
    [ -f "${REGISTRY_ENV_FILE}" ] || return 0
    # `|| true`: a key absent from the file makes grep exit non-zero, and under
    # `set -euo pipefail` that would otherwise abort the whole script silently
    # at the calling assignment (before any output). Swallow it; empty == unset.
    val="$(grep -E "^[[:space:]]*${key}=" "${REGISTRY_ENV_FILE}" 2>/dev/null | tail -1 \
        | cut -d= -f2- \
        | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
              -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'\$//")" || true
    printf '%s' "${val}"
    return 0
}

DOCKERHUB_USER="$(read_env DOCKERHUB_USER)"
DOCKERHUB_TOKEN="$(read_env DOCKERHUB_TOKEN)"
REGISTRY_PORT="${REGISTRY_PORT:-$(read_env XRLENV_REGISTRY_PORT)}"
REGISTRY_PORT="${REGISTRY_PORT:-5010}"
REGISTRY_STORAGE="${REGISTRY_STORAGE:-$(read_env XRLENV_REGISTRY_STORAGE)}"
REGISTRY_STORAGE="${REGISTRY_STORAGE:-/path/to/data${USER}/xrlenv-registry/proxy}"
# We run registry:3 (distribution 3.x). It HONORS proxy.ttl from config.yml
# (registry:2 ignored it and hardcoded 7 days), so retention is config-driven and
# survives a clean redeploy. The config is mounted at /etc/distribution/config.yml
# (the path 3.x reads). registry:3 uses the same /docker/registry/v2 storage
# layout as 2.x, so this is a drop-in image change with no re-warm. Don't set
# REGISTRY_IMAGE back to registry:2 — it reads config from a different path and
# would silently start without the proxy/ttl config.
REGISTRY_IMAGE="${REGISTRY_IMAGE:-registry:3}"
REGISTRY_NAME="${REGISTRY_NAME:-xrlenv-registry-proxy}"
CONFIG="${REGISTRY_CONFIG:-${SCRIPT_DIR}/config.yml}"

echo "==> xrlenv registry proxy"
echo "    env file: ${REGISTRY_ENV_FILE}$( [ -f "${REGISTRY_ENV_FILE}" ] || echo ' (absent)' )"
echo "    port    : ${REGISTRY_PORT}"
echo "    storage : ${REGISTRY_STORAGE}"
echo "    config  : ${CONFIG}"

# Storage dir on FSx + shared-fs guard (symmetric to set_docker_data_root.sh:
# there we *refuse* Lustre for the data-root; here we *want* a shared fs).
mkdir -p "${REGISTRY_STORAGE}"
fstype="$(findmnt -no FSTYPE --target "${REGISTRY_STORAGE}" 2>/dev/null || echo '')"
echo "    storage fstype: ${fstype:-unknown}"
case "${fstype}" in
    lustre|nfs|nfs4) : ;;
    *) echo "WARN: '${REGISTRY_STORAGE}' fstype '${fstype:-unknown}' is not a shared (Lustre/NFS) mount; the cache won't persist on FSx." >&2 ;;
esac

# Upstream auth: map DOCKERHUB_* -> REGISTRY_PROXY_* (distribution reads the
# latter and merges into proxy.*). Passed to the container by *name* (-e VAR,
# no value) so the token isn't on THIS script's command line / process args.
# NOTE: Docker still records container env in Config.Env, so the token IS
# visible via `docker inspect` to anyone with Docker daemon access (which is
# root-equivalent anyway). Treat daemon access as privileged and keep the .env
# file tight (chmod 600). For stronger secrecy, use a registry secret-file
# mechanism instead of env vars.
proxy_env_args=()
if [ -n "${DOCKERHUB_USER}" ] && [ -n "${DOCKERHUB_TOKEN}" ]; then
    export REGISTRY_PROXY_USERNAME="${DOCKERHUB_USER}"
    export REGISTRY_PROXY_PASSWORD="${DOCKERHUB_TOKEN}"
    proxy_env_args=(-e REGISTRY_PROXY_USERNAME -e REGISTRY_PROXY_PASSWORD)
    echo "    upstream auth: DOCKERHUB_USER=${DOCKERHUB_USER} (token read from env file)"
else
    echo "    upstream auth: NONE (anonymous; Docker Hub rate-limits ~100 pulls/6h)."
    echo "                  set DOCKERHUB_USER + DOCKERHUB_TOKEN (a rotated Pro/Team PAT)"
    echo "                  in ${REGISTRY_ENV_FILE} and re-run before warming the full set."
fi

docker pull "${REGISTRY_IMAGE}" >/dev/null
docker rm -f "${REGISTRY_NAME}" >/dev/null 2>&1 || true
docker run -d \
    --name "${REGISTRY_NAME}" \
    --restart=always \
    -p "${REGISTRY_PORT}:5000" \
    -v "${REGISTRY_STORAGE}:/var/lib/registry" \
    -v "${CONFIG}:/etc/distribution/config.yml:ro" \
    "${proxy_env_args[@]}" \
    "${REGISTRY_IMAGE}" >/dev/null
echo "    proxy cache ttl: $(grep -E '^\s*ttl:' "${CONFIG}" | awk '{print $2}') (from config.yml, honored by registry:3)"

echo "==> started ${REGISTRY_NAME}"
sleep 2
docker ps --filter "name=${REGISTRY_NAME}" --format '    {{.Names}} {{.Status}} {{.Ports}}'
if curl -fsS --max-time 5 "http://127.0.0.1:${REGISTRY_PORT}/v2/" >/dev/null; then
    echo "==> /v2/ OK on 127.0.0.1:${REGISTRY_PORT}"
else
    echo "==> /v2/ NOT ready yet — check 'docker logs ${REGISTRY_NAME}'" >&2
fi
echo
echo "Next: on each worker, point Docker at this mirror:"
echo "  sudo MIRROR_URL=http://<this-host-ip>:${REGISTRY_PORT} bash scripts/configure_docker_registry.sh --restart"
