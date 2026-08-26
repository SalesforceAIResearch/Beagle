#!/usr/bin/env bash
# run-registry-scratch.sh — start/restart the xrlenv SCRATCH (build-on-demand)
# registry on the control-plane box. This is the ephemeral, quota-bounded, GC'd
# home for images built on the fly from a template's `image_build:` Dockerfile
# (bring-your-own-Dockerfile). Nodes build the content-addressed ref and
# push it here; peers pull it over the LAN.
#
# Sibling of run-registry-mirror.sh / run-registry-private.sh — keep them straight:
#   - run-registry-mirror.sh   → :5010, PROXY (pull-through cache of docker.io).
#   - run-registry-private.sh → :5011, PRIVATE (durable, operator-pushed).
#   - run-registry-scratch.sh → :5012, SCRATCH (build-on-demand, ephemeral, GC'd).
# All three can run on the same box; they use distinct ports + distinct FSx subdirs.
#
# Keys it reads from the env file (all optional; calling-env values win):
#   XRLENV_SCRATCH_REGISTRY_STORAGE  REQUIRED — shared (FSx/Lustre/NFS) blob-store
#                                   path. No default (fails loud if unset).
#   XRLENV_SCRATCH_REGISTRY_PORT     host port (default 5012)
#   XRLENV_SCRATCH_REGISTRY_HTTP_SECRET  optional shared upload secret (only for a
#                                   multi-replica/LB deploy).
# Override the env-file path with REGISTRY_ENV_FILE=/path/to/.env.
#
# Nodes must list <cp-ip>:5012 in insecure-registries to push/pull over HTTP —
# wire it into bootstrap alongside XRLENV_PRIVATE_REGISTRY, or run
#   sudo PRIVATE_REGISTRY=<cp-ip>:5012 bash deploy/registry/configure_docker_registry.sh --restart
# (the helper takes any host:port to add to insecure-registries).
#
# Usage (on the control-plane box):
#   bash deploy/registry/run-registry-scratch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REGISTRY_ENV_FILE="${REGISTRY_ENV_FILE:-${REPO_ROOT}/.env}"

# Read KEY=value from the env file WITHOUT executing it (same parser as the
# proxy/private runners — the .env is hand-maintained, so parse rather than
# `source`). A value already set in the calling environment wins.
read_env() {
    local key="$1" val=""
    if [ -n "${!key:-}" ]; then printf '%s' "${!key}"; return 0; fi
    [ -f "${REGISTRY_ENV_FILE}" ] || return 0
    val="$(grep -E "^[[:space:]]*${key}=" "${REGISTRY_ENV_FILE}" 2>/dev/null | tail -1 \
        | cut -d= -f2- \
        | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
              -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'\$//")" || true
    printf '%s' "${val}"
    return 0
}

REGISTRY_PORT="${REGISTRY_PORT:-$(read_env XRLENV_SCRATCH_REGISTRY_PORT)}"
REGISTRY_PORT="${REGISTRY_PORT:-5012}"
REGISTRY_STORAGE="${REGISTRY_STORAGE:-$(read_env XRLENV_SCRATCH_REGISTRY_STORAGE)}"
if [ -z "${REGISTRY_STORAGE}" ]; then
    echo "ERROR: XRLENV_SCRATCH_REGISTRY_STORAGE is unset (in ${REGISTRY_ENV_FILE} or the env)." >&2
    echo "       It has NO default — the old /fsx/home/\$USER/... default silently assumed an FSx" >&2
    echo "       layout and is wrong off-cluster. Set it to a shared (FSx/Lustre/NFS) blob-store" >&2
    echo "       path (e.g. /fsx/home/\$USER/xrlenv-registry/scratch) in ${REGISTRY_ENV_FILE}." >&2
    exit 1
fi
# registry:3 for parity with the proxy/private (config at
# /etc/distribution/config.yml). Don't drop to registry:2.
REGISTRY_IMAGE="${REGISTRY_IMAGE:-registry:3}"
REGISTRY_NAME="${REGISTRY_NAME:-xrlenv-registry-scratch}"
CONFIG="${REGISTRY_CONFIG:-${SCRIPT_DIR}/config-scratch.yml}"

echo "==> xrlenv registry (SCRATCH / build-on-demand)"
echo "    env file: ${REGISTRY_ENV_FILE}$( [ -f "${REGISTRY_ENV_FILE}" ] || echo ' (absent)' )"
echo "    port    : ${REGISTRY_PORT}"
echo "    storage : ${REGISTRY_STORAGE}"
echo "    config  : ${CONFIG}"

# Storage dir on FSx + shared-fs guard (symmetric to the private runner: we
# *want* a shared fs so every node pulls the same bytes).
mkdir -p "${REGISTRY_STORAGE}"
fstype="$(findmnt -no FSTYPE --target "${REGISTRY_STORAGE}" 2>/dev/null || echo '')"
echo "    storage fstype: ${fstype:-unknown}"
case "${fstype}" in
    lustre|nfs|nfs4) : ;;
    *) echo "WARN: '${REGISTRY_STORAGE}' fstype '${fstype:-unknown}' is not a shared (Lustre/NFS) mount; pushed images won't be visible cluster-wide." >&2 ;;
esac

# Optional shared upload secret (only matters for a multi-replica/LB deploy).
HTTP_SECRET="${HTTP_SECRET:-$(read_env XRLENV_SCRATCH_REGISTRY_HTTP_SECRET)}"
secret_env_args=()
if [ -n "${HTTP_SECRET}" ]; then
    export REGISTRY_HTTP_SECRET="${HTTP_SECRET}"
    secret_env_args=(-e REGISTRY_HTTP_SECRET)
    echo "    http secret: set (multi-replica safe)"
fi

docker pull "${REGISTRY_IMAGE}" >/dev/null
docker rm -f "${REGISTRY_NAME}" >/dev/null 2>&1 || true
docker run -d \
    --name "${REGISTRY_NAME}" \
    --restart=always \
    -p "${REGISTRY_PORT}:5000" \
    -v "${REGISTRY_STORAGE}:/var/lib/registry" \
    -v "${CONFIG}:/etc/distribution/config.yml:ro" \
    "${secret_env_args[@]}" \
    "${REGISTRY_IMAGE}" >/dev/null

echo "==> started ${REGISTRY_NAME}"
sleep 2
docker ps --filter "name=${REGISTRY_NAME}" --format '    {{.Names}} {{.Status}} {{.Ports}}'
if curl -fsS --max-time 5 "http://127.0.0.1:${REGISTRY_PORT}/v2/" >/dev/null; then
    echo "==> /v2/ OK on 127.0.0.1:${REGISTRY_PORT}"
else
    echo "==> /v2/ NOT ready yet — check 'docker logs ${REGISTRY_NAME}'" >&2
fi
echo
echo "Next:"
echo "  1. Point the control plane at this registry so image_build templates"
echo "     build into it:"
echo "       XRLENV_SCRATCH_REGISTRY_HOST=<this-host-ip> XRLENV_SCRATCH_REGISTRY_PORT=${REGISTRY_PORT}"
echo "     (set in the CP's env; the coordinator forms <host>:<port> refs)."
echo "  2. Let nodes reach this HTTP registry (insecure-registries), like the"
echo "     private registry — via bootstrap or configure_docker_registry.sh."
echo "  3. Schedule the GC so the store stays bounded (TTL + per-namespace quota,"
echo "     active-run digests exempt):"
echo "       python deploy/registry/scratch_registry_gc.py --registry 127.0.0.1:${REGISTRY_PORT} \\"
echo "         --ttl \${XRLENV_SCRATCH_GC_TTL:-72h} --quota-gb \${XRLENV_SCRATCH_REGISTRY_QUOTA_GB:-500}"
