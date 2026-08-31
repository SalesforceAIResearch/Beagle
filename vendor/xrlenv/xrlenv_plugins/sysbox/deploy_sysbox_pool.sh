#!/usr/bin/env bash
# deploy_sysbox_pool.sh — install Sysbox on every node marked ``sysbox: true`` in
# nodes.yaml, using the checksum-pinned vendored binaries from build_sysbox.sh.
#
# This is the "wire the vendored-binary install for a nodes.yaml-declared sysbox
# pool" entry point. It reads the pool from nodes.yaml, then ssh'es
# install_sysbox_node.sh onto each pool node. Because Sysbox is a
# container-escape surface, this deliberately does NOT run as part of the normal
# per-node bootstrap — it is a separate, explicit, operator-invoked step scoped
# to the declared pool.
#
# Usage:
#   bash xrlenv_plugins/sysbox/deploy_sysbox_pool.sh [nodes.yaml] [VENDOR_DIR]
#
# Prereqs:
#   - build_sysbox.sh has produced VENDOR_DIR (default:
#     ${SYSBOX_VENDOR_ROOT}/<pinned-commit>, resolved per cluster by pin.env).
#     The vendor root is per-cluster storage, so run the build once per cluster.
#   - passwordless ssh + sudo to each pool node's address.
#   - /fsx (or wherever this repo lives) is reachable on the nodes, OR the vendor
#     dir is copied to each node (this script scp's it when the repo path differs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=./pin.env
source "${SCRIPT_DIR}/pin.env"

NODES_YAML="${1:-${REPO_ROOT}/nodes.yaml}"
VENDOR_DIR="${2:-${SYSBOX_VENDOR_ROOT}/${SYSBOX_RUNC_COMMIT}}"

log()  { printf '==> %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f "${NODES_YAML}" ]] || die "nodes.yaml not found at ${NODES_YAML}"
[[ -f "${VENDOR_DIR}/SHA256SUMS" ]] || die \
    "no vendored binaries at ${VENDOR_DIR}; run build_sysbox.sh first"

# Extract the sysbox pool (id + address) via the project venv + the nodes_yaml
# loader — the single parser, so this can't drift from what the control plane
# reads.
PY="${REPO_ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="python3"
mapfile -t POOL < <("${PY}" - "${NODES_YAML}" <<'PYEOF'
import sys
from pathlib import Path
from xrlenv.control.nodes_yaml import load_nodes_yaml
inv = load_nodes_yaml(Path(sys.argv[1]))
for n in inv.sysbox_pool():
    print(f"{n.id}\t{n.address or n.id}")
PYEOF
)

[[ "${#POOL[@]}" -gt 0 ]] || die \
    "no nodes marked 'sysbox: true' in ${NODES_YAML}. Add 'sysbox: true' to the \
pool nodes (see README.md)."

log "sysbox pool (${#POOL[@]} node(s)):"
printf '    %s\n' "${POOL[@]}"

# Optional: warn if the policy doesn't permit the runtime — the install alone
# won't let any acquire request sysbox until allowed_runtimes includes it.
"${PY}" - "${NODES_YAML}" <<'PYEOF' || true
import sys
from pathlib import Path
from xrlenv.control.nodes_yaml import load_nodes_yaml
inv = load_nodes_yaml(Path(sys.argv[1]))
if "sysbox-runc" not in inv.policy.allowed_runtimes:
    print("==> WARNING: policy.allowed_runtimes does not include 'sysbox-runc'; "
          "no acquire can request it until you add it and restart the control plane.")
PYEOF

for entry in "${POOL[@]}"; do
    node_id="${entry%%$'\t'*}"
    addr="${entry##*$'\t'}"
    log "installing Sysbox on ${node_id} (${addr})"
    # Copy the vendored set + scripts to a temp dir on the node, then run the
    # installer there (robust whether or not the node shares this repo's fs).
    remote_tmp="/tmp/xrlenv-sysbox-install"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${addr}" \
        "rm -rf ${remote_tmp} && mkdir -p ${remote_tmp}/vendor"
    scp -q -o BatchMode=yes -o StrictHostKeyChecking=no \
        "${SCRIPT_DIR}/pin.env" "${SCRIPT_DIR}/install_sysbox_node.sh" \
        "${addr}:${remote_tmp}/"
    scp -q -o BatchMode=yes -o StrictHostKeyChecking=no \
        "${VENDOR_DIR}"/* "${addr}:${remote_tmp}/vendor/"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${addr}" \
        "sudo bash ${remote_tmp}/install_sysbox_node.sh ${remote_tmp}/vendor" \
        || die "install failed on ${node_id} (${addr})"
    log "OK: ${node_id}"
done

log "sysbox pool deploy complete (${#POOL[@]} node(s))."
log "Ensure nodes.yaml policy.allowed_runtimes includes 'sysbox-runc', then"
log "restart the control plane so the KwargsPolicy picks it up."
