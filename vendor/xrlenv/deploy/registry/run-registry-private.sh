#!/usr/bin/env bash
# run-registry-private.sh — start/restart the xrlenv PRIVATE (writable) registry on
# the control-plane box: a `docker push`-able registry that holds images we build
# ourselves from Dockerfiles (e.g. camel-ai/seta-env tasks), backed by FSx.
#
# It is the SIBLING of run-registry-mirror.sh — keep the two straight:
#   - run-registry-mirror.sh → :5010, PROXY (pull-through cache of docker.io).
#                             Cannot be pushed to. Storage ~/xrlenv-registry/proxy.
#   - run-registry-private.sh → :5011, PRIVATE (writable). Push your built images
#                             here. Storage ~/xrlenv-registry/private.
# Both can run on the same box; they use distinct ports + distinct FSx subdirs.
#
# Keys it reads from the env file (all optional; calling-env values win):
#   XRLENV_PRIVATE_REGISTRY_STORAGE      REQUIRED — shared (FSx/Lustre/NFS)
#                                      blob-store path. No default (fails if unset).
#   XRLENV_PRIVATE_REGISTRY_PORT         host port (default 5011)
#   XRLENV_PRIVATE_REGISTRY_HTTP_SECRET  optional shared upload secret. A single
#                                      instance doesn't need one (registry just
#                                      generates a random secret + logs a benign
#                                      warning). Set a stable value ONLY if you
#                                      ever run 2+ replicas behind a load balancer
#                                      so an in-flight upload routed to a different
#                                      replica still validates.
# Override the env-file path with REGISTRY_ENV_FILE=/path/to/.env.
#
# No upstream Docker Hub auth is needed here (this is not a proxy — it never talks
# to Docker Hub). Build hosts pull their FROM base images through the :5010 proxy.
#
# Usage (on the control-plane box):
#   bash deploy/registry/run-registry-private.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REGISTRY_ENV_FILE="${REGISTRY_ENV_FILE:-${REPO_ROOT}/.env}"

# Read KEY=value from the env file WITHOUT executing it (same parser as
# run-registry-mirror.sh — the .env is hand-maintained, so parse rather than
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

REGISTRY_PORT="${REGISTRY_PORT:-$(read_env XRLENV_PRIVATE_REGISTRY_PORT)}"
REGISTRY_PORT="${REGISTRY_PORT:-5011}"
REGISTRY_STORAGE="${REGISTRY_STORAGE:-$(read_env XRLENV_PRIVATE_REGISTRY_STORAGE)}"
if [ -z "${REGISTRY_STORAGE}" ]; then
    echo "ERROR: XRLENV_PRIVATE_REGISTRY_STORAGE is unset (in ${REGISTRY_ENV_FILE} or the env)." >&2
    echo "       It has NO default — the old /fsx/home/\$USER/... default silently assumed an FSx" >&2
    echo "       layout and is wrong off-cluster. Set it to a shared (FSx/Lustre/NFS) blob-store" >&2
    echo "       path (e.g. /fsx/home/\$USER/xrlenv-registry/private) in ${REGISTRY_ENV_FILE}." >&2
    exit 1
fi
# registry:3 for parity with the proxy (same storage layout, config at
# /etc/distribution/config.yml). Don't drop to registry:2 — it reads config from
# a different path and would start with defaults (no health checks etc.).
REGISTRY_IMAGE="${REGISTRY_IMAGE:-registry:3}"
REGISTRY_NAME="${REGISTRY_NAME:-xrlenv-registry-private}"
CONFIG="${REGISTRY_CONFIG:-${SCRIPT_DIR}/config-private.yml}"

echo "==> xrlenv registry (PRIVATE / writable)"
echo "    env file: ${REGISTRY_ENV_FILE}$( [ -f "${REGISTRY_ENV_FILE}" ] || echo ' (absent)' )"
echo "    port    : ${REGISTRY_PORT}"
echo "    storage : ${REGISTRY_STORAGE}"
echo "    config  : ${CONFIG}"

# Storage dir on FSx + shared-fs guard (symmetric to run-registry-mirror.sh: we
# *want* a shared fs so every host pulls the same bytes).
mkdir -p "${REGISTRY_STORAGE}"
fstype="$(findmnt -no FSTYPE --target "${REGISTRY_STORAGE}" 2>/dev/null || echo '')"
echo "    storage fstype: ${fstype:-unknown}"
case "${fstype}" in
    lustre|nfs|nfs4) : ;;
    *) echo "WARN: '${REGISTRY_STORAGE}' fstype '${fstype:-unknown}' is not a shared (Lustre/NFS) mount; pushed images won't be visible cluster-wide." >&2 ;;
esac

# Optional shared upload secret (only matters for a multi-replica/LB deploy).
# Passed by name (-e VAR, no value) so it isn't on this script's process args.
HTTP_SECRET="${HTTP_SECRET:-$(read_env XRLENV_PRIVATE_REGISTRY_HTTP_SECRET)}"
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
echo "  1. Let nodes reach this HTTP registry. Normally automatic: set"
echo "       XRLENV_PRIVATE_REGISTRY=<this-host-ip>:${REGISTRY_PORT}"
echo "     in your node bring-up (bootstrap wires it into insecure-registries, like"
echo "     XRLENV_REGISTRY_MIRROR). To fix an already-running node by hand:"
echo "       sudo PRIVATE_REGISTRY=<this-host-ip>:${REGISTRY_PORT} bash deploy/registry/configure_docker_registry.sh --restart"
echo "  2. Build + push a plan's images across the fleet (native — no Slurm):"
echo "       xrlenv build push --plan <build_plan.yaml> \\"
echo "         --registry <this-host-ip>:${REGISTRY_PORT} --connect-host <admin-host>"
echo "     or a single shard on one build host:"
echo "       python deploy/registry/build_and_push_images.py --plan <build_plan.yaml> \\"
echo "         --registry <this-host-ip>:${REGISTRY_PORT}"
