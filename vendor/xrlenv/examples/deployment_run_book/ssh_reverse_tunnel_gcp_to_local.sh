#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
echo "REPO_ROOT: ${REPO_ROOT}"
source "${REPO_ROOT}/.env"

GCP_VM_NAME="${1:-${GCP_VM_NAME}}"
GCP_ZONE="${GCP_ZONE}"
GCP_PROJECT_ID="${GCP_PROJECT_ID}"
REMOTE_FORWARD_PORTS=(50051 9090 8080)

echo "GCP_VM_NAME: ${GCP_VM_NAME}"
echo "GCP_ZONE: ${GCP_ZONE}"
echo "GCP_PROJECT_ID: ${GCP_PROJECT_ID}"

# error out if the GCP_VM_NAME, GCP_ZONE, or GCP_PROJECT_ID is not set
if [ -z "${GCP_VM_NAME}" ] || [ -z "${GCP_ZONE}" ] || [ -z "${GCP_PROJECT_ID}" ]; then
    echo "Usage: $0 <GCP_VM_NAME>"
    echo "GCP_ZONE and GCP_PROJECT_ID must be set in ${REPO_ROOT}/.env"
    exit 1
fi

REMOTE_FORWARD_FLAGS=()
REMOTE_CLEANUP_TARGETS=()
for port in "${REMOTE_FORWARD_PORTS[@]}"; do
    REMOTE_FORWARD_FLAGS+=(--ssh-flag="-R ${port}:127.0.0.1:${port}")
    REMOTE_CLEANUP_TARGETS+=("${port}/tcp")
done

if command -v autossh >/dev/null 2>&1; then
    SSH_CMD=(autossh -M 0)
    export AUTOSSH_GATETIME="${AUTOSSH_GATETIME:-0}"
else
    echo "autossh not found; falling back to plain ssh (no auto-reconnect if the link drops; brew install autossh to get reconnects)" >&2
    SSH_CMD=(ssh)
fi

if [ "${CLEAN_STALE_REMOTE_PORTS:-1}" != "0" ]; then
    echo "Cleaning stale remote listeners on ${GCP_VM_NAME}: ${REMOTE_CLEANUP_TARGETS[*]}"
    gcloud compute ssh "${GCP_VM_NAME}" \
      --zone "${GCP_ZONE}" \
      --project "${GCP_PROJECT_ID}" \
      --tunnel-through-iap \
      --command="if command -v fuser >/dev/null 2>&1; then sudo fuser -k ${REMOTE_CLEANUP_TARGETS[*]} >/dev/null 2>&1 || true; else echo 'warning: fuser not found; skipping stale port cleanup' >&2; fi"
fi

# run on the local machine
SSH_COMMAND="$(gcloud compute ssh "${GCP_VM_NAME}" \
  --zone "${GCP_ZONE}" \
  --project "${GCP_PROJECT_ID}" \
  --tunnel-through-iap \
  --dry-run \
  --ssh-flag="-N" \
  --ssh-flag="-o ExitOnForwardFailure=yes" \
  --ssh-flag="-o ServerAliveInterval=30" \
  --ssh-flag="-o ServerAliveCountMax=3" \
  "${REMOTE_FORWARD_FLAGS[@]}")"

# gcloud emits a shell-escaped ssh command; reuse its args under autossh.
eval "set -- ${SSH_COMMAND}"
shift

echo "Starting reverse tunnel to ${GCP_VM_NAME}"
exec "${SSH_CMD[@]}" "$@"