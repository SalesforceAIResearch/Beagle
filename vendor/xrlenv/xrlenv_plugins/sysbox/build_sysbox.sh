#!/usr/bin/env bash
# build_sysbox.sh — build the patched Sysbox static binaries from source and
# emit a checksummed vendor set for install_sysbox_node.sh to consume.
#
# WHY THIS EXISTS: the packaged ``sysbox-ce`` 0.7.0 release cannot run any
# container on our Docker 29.x / containerd 2.x nodes (fails at task-create with
# ``namespace {"time" ""} does not exist`` — upstream nestybox/sysbox#1011). The
# fix (sysbox-runc PR #106) is merged to sysbox-runc ``main`` but not in any
# packaged release, so we build it from source and vendor a checksum-pinned
# binary. See README.md for the full story + security posture.
#
# This runs the OFFICIAL containerized Sysbox build (``make sysbox-static``),
# which compiles inside a ``--runtime=runc`` build container — the host is not
# polluted with a Go toolchain. Requires: docker, git, ~2 GB disk, internet to
# GitHub + the docker.io build base (routed through your registry mirror if the
# host has one configured).
#
# Usage:
#   bash xrlenv_plugins/sysbox/build_sysbox.sh [OUT_DIR]
# Default OUT_DIR: ${SYSBOX_VENDOR_ROOT}/<commit>/ (resolved per cluster by
# pin.env; the vendor root is per-cluster storage, so run this once per cluster)
#
# Output: OUT_DIR/{sysbox-runc,sysbox-mgr,sysbox-fs}, OUT_DIR/SHA256SUMS, and
# OUT_DIR/PROVENANCE.txt. Commit those (or host them) so install_sysbox_node.sh
# can verify + overlay them without rebuilding on every node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./pin.env
source "${SCRIPT_DIR}/pin.env"

OUT_DIR="${1:-${SYSBOX_VENDOR_ROOT}/${SYSBOX_RUNC_COMMIT}}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

log()  { printf '==> %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker is required (the build runs in a container)"
command -v git    >/dev/null 2>&1 || die "git is required"

# Clone the PARENT at a pinned commit, then let it place its own submodules.
# ``--filter=blob:none`` keeps full commit history (needed to check out an
# arbitrary pinned commit) while deferring blob download, so this stays about as
# fast as the old ``--depth 1``.
#
# Do NOT go back to cloning the default branch: the parent supplies the build
# toolchain and records sysbox-fs / -mgr / -libs / -ipc. Floating the parent
# while pinning only sysbox-runc mixes eras — the newer siblings' go.mod pulls a
# newer Go toolchain that then rejects the pinned runc's go.mod with "updates to
# go.mod needed" (the 2026-08-08 build failure).
log "cloning nestybox/sysbox into ${WORK_DIR}/sysbox (parent pin ${SYSBOX_PARENT_COMMIT})"
git clone --filter=blob:none https://github.com/nestybox/sysbox.git "${WORK_DIR}/sysbox"
git -C "${WORK_DIR}/sysbox" checkout --quiet "${SYSBOX_PARENT_COMMIT}" \
    || die "could not checkout nestybox/sysbox ${SYSBOX_PARENT_COMMIT} (SYSBOX_PARENT_COMMIT in pin.env)"
log "initializing submodules recorded by ${SYSBOX_PARENT_COMMIT}"
git -C "${WORK_DIR}/sysbox" submodule update --init --recursive

# The parent commit RECORDS the whole submodule set, so sysbox-runc lands on the
# pinned commit with no forced checkout. Assert that rather than override it: if
# the two pins disagree it is a config error to fix in pin.env, not something to
# paper over by dragging one submodule out of step with its siblings.
_actual_commit="$(git -C "${WORK_DIR}/sysbox/sysbox-runc" rev-parse --short=8 HEAD)"
[[ "${_actual_commit}" == "${SYSBOX_RUNC_COMMIT}" ]] || die \
    "parent ${SYSBOX_PARENT_COMMIT} records sysbox-runc ${_actual_commit}, but pin.env \
pins SYSBOX_RUNC_COMMIT=${SYSBOX_RUNC_COMMIT}. Update pin.env so the two agree \
(pick the newest parent commit whose recorded sysbox-runc is the one you want)."
log "sysbox-runc at ${_actual_commit}: $(git -C "${WORK_DIR}/sysbox/sysbox-runc" log --oneline -1)"

log "building static binaries (containerized 'make sysbox-static'; ~15-30 min, first run pulls a builder image)"
make -C "${WORK_DIR}/sysbox" sysbox-static

# The static targets land under each submodule's build/<arch>/ tree.
ARCH="$(uname -m)"; case "${ARCH}" in x86_64) ARCH=amd64 ;; aarch64) ARCH=arm64 ;; esac
declare -A SRC=(
    [sysbox-runc]="${WORK_DIR}/sysbox/sysbox-runc/build/${ARCH}/sysbox-runc"
    [sysbox-mgr]="${WORK_DIR}/sysbox/sysbox-mgr/build/${ARCH}/sysbox-mgr"
    [sysbox-fs]="${WORK_DIR}/sysbox/sysbox-fs/build/${ARCH}/sysbox-fs"
)

mkdir -p "${OUT_DIR}"
for b in ${SYSBOX_BINARIES}; do
    [[ -x "${SRC[$b]}" ]] || die "expected build output missing: ${SRC[$b]}"
    install -m 0755 "${SRC[$b]}" "${OUT_DIR}/${b}"
done

log "generating SHA256SUMS"
( cd "${OUT_DIR}" && sha256sum ${SYSBOX_BINARIES} > SHA256SUMS )

cat > "${OUT_DIR}/PROVENANCE.txt" <<EOF
Patched Sysbox static binaries — vendored for the xrlenv sysbox node pool.
Built from https://github.com/nestybox/sysbox (recursive) via 'make sysbox-static'.
sysbox-runc commit: ${_actual_commit} (pin: ${SYSBOX_RUNC_COMMIT})
Carries the fix for nestybox/sysbox#1011 (packaged sysbox-ce ${SYSBOX_CE_VERSION}
fails 'namespace {"time" ""} does not exist' on Docker 29.x / containerd 2.x) —
sysbox-runc PR #106. Overlay these over a packaged sysbox-ce ${SYSBOX_CE_VERSION}
install (which supplies the systemd units, sysctls, and 'sysbox' user) via
install_sysbox_node.sh. Verify against SHA256SUMS before installing.
Arch: ${ARCH}.
EOF

log "done. vendored set at: ${OUT_DIR}"
cat "${OUT_DIR}/SHA256SUMS"
echo
log "next: run install_sysbox_node.sh on each sysbox-pool node (see README.md)"
