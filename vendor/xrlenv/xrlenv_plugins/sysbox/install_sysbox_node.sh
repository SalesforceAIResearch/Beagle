#!/usr/bin/env bash
# install_sysbox_node.sh — install Sysbox on ONE node of the sysbox pool, using
# the checksum-pinned patched binaries from build_sysbox.sh.
#
# Run ON the target node (or via ssh <node> 'sudo bash -s' < this-file). Requires
# root/sudo. Idempotent + safe to re-run. This is the operator-gated, dev/
# single-tenant install path — do NOT run it on a shared multi-tenant node
# (a Sysbox container's inner root can rewrite its own netns, so xrlenv's
# post-install egress allowlist is not a trusted boundary for it — see the
# xrlenv core §6 egress handling and README.md "Security").
#
# What it does, in the order the 2026-07-05 de-risk validated:
#   1. Install the packaged sysbox-ce (for its systemd units, sysctls, 'sysbox'
#      subuid/subgid user, and non-destructive daemon.json runtime merge).
#   2. Overlay the checksum-verified PATCHED binaries (the packaged ones can't
#      run on Docker 29.x/containerd 2.x — nestybox/sysbox#1011).
#   3. Assert docker's default-runtime is unset-or-runc (else allowed_runtimes
#      is silently bypassed — see xrlenv core §5.1/§9).
#   4. Restart sysbox + docker, then bounce xrlenv-node (it is bound to
#      docker.service and does NOT auto-revive on a docker restart).
#
# Usage:
#   sudo bash install_sysbox_node.sh [VENDOR_DIR]
# VENDOR_DIR default: ${SYSBOX_VENDOR_ROOT}/<pinned-commit>/, where
# SYSBOX_VENDOR_ROOT is resolved per cluster by pin.env (must contain the three
# binaries + SHA256SUMS from build_sysbox.sh — run it once per cluster).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./pin.env
source "${SCRIPT_DIR}/pin.env"

VENDOR_DIR="${1:-${SYSBOX_VENDOR_ROOT}/${SYSBOX_RUNC_COMMIT}}"
DEB_URL="https://github.com/nestybox/sysbox/releases/download/v${SYSBOX_CE_VERSION}/sysbox-ce_${SYSBOX_CE_VERSION}.linux_amd64.deb"

log()  { printf '==> %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(id -u)" -eq 0 ]] || die "must run as root (use sudo)"
command -v docker >/dev/null 2>&1 || die "docker must be installed first (bootstrap runs before this)"

# ── 0) verify the vendored binaries against SHA256SUMS ─────────────────────────
[[ -f "${VENDOR_DIR}/SHA256SUMS" ]] || die "no SHA256SUMS in ${VENDOR_DIR}; run build_sysbox.sh first"
log "verifying vendored binaries against SHA256SUMS"
( cd "${VENDOR_DIR}" && sha256sum -c SHA256SUMS ) || die "checksum verification FAILED — refusing to install an unverified runtime"

# ── 1) install packaged sysbox-ce (units/sysctls/user) ────────────────────────
if ! dpkg -l sysbox-ce 2>/dev/null | grep -q '^ii'; then
    log "installing packaged sysbox-ce ${SYSBOX_CE_VERSION} (units/sysctls/user)"
    _deb="$(mktemp --suffix=.deb)"
    curl -fSL -o "${_deb}" "${DEB_URL}"
    # The .deb postinst refuses to auto-configure while docker has running
    # containers (it wants a clean network reconfig). Work around it exactly as
    # the de-risk did: stop docker, let the postinst write config with docker
    # down (it then skips the restart), start docker back up afterward.
    systemctl stop docker.socket docker.service 2>/dev/null || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${_deb}" 2>/dev/null || true
    DEBIAN_FRONTEND=noninteractive dpkg --configure -a
    rm -f "${_deb}"
else
    log "sysbox-ce already installed; stopping docker + sysbox for the binary overlay"
    systemctl stop docker.socket docker.service 2>/dev/null || true
fi

# ── 2) overlay the checksum-verified patched binaries ─────────────────────────
systemctl stop sysbox.service sysbox-fs.service sysbox-mgr.service 2>/dev/null || true
log "overlaying patched binaries from ${VENDOR_DIR}"
for b in ${SYSBOX_BINARIES}; do
    install -m 0755 "${VENDOR_DIR}/${b}" "/usr/bin/${b}"
done

# ── 3) assert docker default-runtime is unset or runc ─────────────────────────
# The sysbox postinst adds runtimes.sysbox-runc as an *available* runtime, not
# the default. If default-runtime became sysbox-runc, every acquire silently
# gets it, bypassing allowed_runtimes. Accept "unset" (docker treats an absent
# key as runc).
_default_rt="$(jq -r '."default-runtime" // "runc"' /etc/docker/daemon.json 2>/dev/null || echo runc)"
[[ "${_default_rt}" == "runc" ]] || die \
    "docker default-runtime is ${_default_rt@Q} (expected unset or 'runc'). \
Fix /etc/docker/daemon.json (remove default-runtime or set it to runc) and re-run."

# ── 4) start sysbox + docker, then bounce xrlenv-node ─────────────────────────
log "starting sysbox + docker"
systemctl start sysbox.service
systemctl start docker.service
# xrlenv-node.service is bound to docker.service: a docker restart stops it and
# it is NOT auto-revived. Start it explicitly so the node re-registers.
if systemctl list-unit-files | grep -q '^xrlenv-node.service'; then
    log "restarting xrlenv-node (bound to docker.service)"
    systemctl start xrlenv-node.service || true
fi

# ── verify ────────────────────────────────────────────────────────────────────
log "verifying"
docker info 2>/dev/null | grep -qi 'sysbox-runc' \
    && log "OK: docker advertises the sysbox-runc runtime" \
    || die "sysbox-runc not registered in docker after install"
log "OK: default-runtime=$(docker info --format '{{.DefaultRuntime}}' 2>/dev/null)"
log "OK: sysbox-runc version: $(sysbox-runc --version 2>/dev/null | awk '/version/{print $2; exit}')"
log "done. This node is now in the sysbox pool. Add it to nodes.yaml's sysbox"
log "  pool (see README.md) so the control plane advertises + schedules it."
