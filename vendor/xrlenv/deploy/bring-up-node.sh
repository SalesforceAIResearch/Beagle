#!/usr/bin/env bash
# bring-up-node.sh — one-shot bring-up for a freshly provisioned VM.
#
# Auto-detects the cloud (GCP / AWS), runs the right cloud-specific
# bootstrap, wires the node token into a systemd drop-in, and tails
# the daemon's first few seconds of logs so the operator sees the
# control-plane connect happen (or fails fast with a useful error).
#
# Usage::
#
#     # On the laptop / control plane (one-time per node):
#     xrlenv tokens issue node                # → prints a token
#
#     # On the new VM:
#     git clone <your-fork>/XRLEnv.git ~/xrlenv && cd ~/xrlenv
#     export XRLENV_CONTROL_PLANE=control.example.com:50051
#     export XRLENV_NODE_TOKEN=<paste-the-token-from-step-above>
#     sudo -E bash deploy/bring-up-node.sh
#
# The script collapses what was previously: clone → set vars → run
# cloud-specific bootstrap → ``systemctl edit xrlenv-node`` to paste
# the token → ``systemctl restart xrlenv-node`` → ``journalctl -u
# xrlenv-node -f`` to confirm.
#
# Required env vars (must survive ``sudo -E``):
#   XRLENV_CONTROL_PLANE   <host:port> of the control plane gRPC port
#   XRLENV_NODE_TOKEN      bearer token issued by ``xrlenv tokens issue node``
#                          (omit for an unauthenticated phase-0 smoke)
#
# Optional:
#   XRLENV_NODE_ID         stable id; auto-detected from cloud metadata
#   XRLENV_REPO            absolute path to the xrlenv checkout (defaults
#                          to the directory two levels up from this script)
#   XRLENV_FORCE_CLOUD     "gcp" or "aws" to skip auto-detect (rarely needed)
#   XRLENV_LOG_TAIL_S      seconds to tail journalctl after bring-up (default 8)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_preflight.sh
source "${SCRIPT_DIR}/_preflight.sh"

require_env XRLENV_CONTROL_PLANE \
    "Pass XRLENV_CONTROL_PLANE=<host>:<port> and re-run with sudo -E."

if [[ -z "${XRLENV_NODE_TOKEN:-}" ]]; then
    log "WARN: XRLENV_NODE_TOKEN not set — skipping token drop-in."
    log "      The daemon will connect without a bearer token; the control plane"
    log "      will reject it unless it's running in unauth (phase-0 smoke) mode."
fi

# Default ``XRLENV_REPO`` to the checkout this script lives in. Saves
# the operator one ``export`` if they're already cd'd into the repo.
if [[ -z "${XRLENV_REPO:-}" ]]; then
    export XRLENV_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ ! -f "${XRLENV_REPO}/pyproject.toml" ]]; then
    die "XRLENV_REPO=${XRLENV_REPO} doesn't look like an xrlenv checkout"
fi

# Cloud detection. GCP's metadata service requires a header; AWS's
# IMDSv2 requires a PUT for a token first. We try GCP first since its
# probe is cheaper and unambiguous.
detect_cloud() {
    if [[ -n "${XRLENV_FORCE_CLOUD:-}" ]]; then
        echo "${XRLENV_FORCE_CLOUD}"
        return 0
    fi
    if curl -fsS -m 2 -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/id \
        >/dev/null 2>&1; then
        echo "gcp"
        return 0
    fi
    if curl -fsS -m 2 -X PUT \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
        http://169.254.169.254/latest/api/token \
        >/dev/null 2>&1; then
        echo "aws"
        return 0
    fi
    echo ""
    return 1
}

CLOUD="$(detect_cloud)" || true
if [[ -z "${CLOUD}" ]]; then
    die "could not detect cloud provider via metadata services. \
Set XRLENV_FORCE_CLOUD=gcp or =aws and re-run."
fi
log "detected cloud: ${CLOUD}"

case "${CLOUD}" in
    gcp)
        bootstrap_script="${SCRIPT_DIR}/bootstrap-gcp.sh"
        ;;
    aws)
        bootstrap_script="${SCRIPT_DIR}/bootstrap-aws.sh"
        ;;
    *)
        die "unsupported cloud '${CLOUD}'. XRLENV_FORCE_CLOUD must be 'gcp' or 'aws'."
        ;;
esac

if [[ ! -x "${bootstrap_script}" && ! -r "${bootstrap_script}" ]]; then
    die "bootstrap script ${bootstrap_script} not found / readable"
fi

log "running ${bootstrap_script} (control plane: ${XRLENV_CONTROL_PLANE})"
bash "${bootstrap_script}"

# Tail the daemon's first few seconds of journalctl so the operator
# sees the gRPC connect or a clear failure mode without a separate
# ``journalctl -u xrlenv-node -f`` step.
tail_seconds="${XRLENV_LOG_TAIL_S:-8}"
log ""
log "tailing journalctl -u xrlenv-node for ${tail_seconds}s — look for"
log "  'connected' (success), 'auth failed' (bad token), or"
log "  'control plane unreachable' (network / firewall)..."
log ""
journalctl -u xrlenv-node --since "${tail_seconds} seconds ago" --no-pager \
    --output=cat || true
log ""
log "bring-up complete. Verify on the control plane with:"
log "  xrlenv nodes"
log ""
log "Continue tailing with:"
log "  journalctl -u xrlenv-node -f"
