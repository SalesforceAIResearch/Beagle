#!/usr/bin/env bash
# refresh.sh — fast path for "I just `git pull`-ed on this VM, push the
# changes into the running xrlenv-node service".
#
# Reinstalls the xrlenv package into the existing venv (non-editable,
# matching the bootstrap path) AND re-runs the additive directory /
# node.env config so a release that adds a new ``XRLENV_*`` env var
# or a new writable subdirectory under ``/var/cache/xrlenv/`` lands
# correctly without requiring a full bootstrap re-run. Operators
# don't need to mentally distinguish "source-only refresh" vs
# "source + new config knob" — refresh.sh handles both.
#
# Skips the heavyweight bootstrap steps (user creation, system
# package install, Python interpreter install) because none of
# that changes on a normal release.
#
# Usage:
#   sudo -E bash deploy/refresh.sh             # uses the script's repo root
#   sudo -E bash deploy/refresh.sh /path/to/xrlenv
#   XRLENV_REPO=/path sudo -E bash deploy/refresh.sh
#
#   # Rotate Docker Hub auth at refresh time (new since 2026-05-12):
#   sudo DOCKERHUB_USER=<handle> DOCKERHUB_TOKEN=<dckr_pat_...> \
#       bash deploy/refresh.sh
#   # Rewrites /opt/xrlenv/.docker/config.json before the daemon
#   # restart so docker-py picks up the new PAT on next APIClient
#   # construction. Omit the vars to preserve existing auth.
#
# Exit codes:
#   0 — refresh complete; service running
#   2 — pre-flight failed (no venv, no pyproject.toml, etc.) — fix and re-run
#   3 — pip install failed
#   4 — service failed to start after restart

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_preflight.sh
source "${SCRIPT_DIR}/_preflight.sh"
# bootstrap-common.sh is side-effect-free at source time (the
# require_env checks only fire from validate_required_env_for_bootstrap,
# which we don't call here). We borrow ensure_directories +
# install_systemd_unit to keep config drift between bootstrap-managed
# and refresh-managed nodes from creeping in.
# shellcheck source=./bootstrap-common.sh
source "${SCRIPT_DIR}/bootstrap-common.sh"

VENV_PYTHON="${INSTALL_ROOT}/.venv/bin/python"
VENV_PIP="${INSTALL_ROOT}/.venv/bin/pip"
VENV_BIN="${INSTALL_ROOT}/.venv/bin/xrlenv-node"
SERVICE="xrlenv-node.service"

# Pick the repo root. Priority: positional arg > XRLENV_REPO env > script's
# parent-of-parent (the repo when this script is at deploy/refresh.sh).
if (( $# >= 1 )); then
    REPO="$(cd "$1" && pwd)"
elif [[ -n "${XRLENV_REPO:-}" ]]; then
    REPO="$(cd "${XRLENV_REPO}" && pwd)"
else
    REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# ─── Pre-flight ──────────────────────────────────────────────────────────────

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    die "refresh.sh must run as root (the venv is owned by the xrlenv user; \
restart and pip install need privilege). Use 'sudo -E bash deploy/refresh.sh'."
fi

if [[ ! -f "${REPO}/pyproject.toml" ]]; then
    die "no pyproject.toml under ${REPO}. \
Pass the repo root as the first arg, set XRLENV_REPO=<path>, or run this \
script from inside the xrlenv checkout."
fi

if [[ ! -x "${VENV_PYTHON}" || ! -x "${VENV_PIP}" ]]; then
    die "no Python venv at ${INSTALL_ROOT}/.venv. \
This VM hasn't been bootstrapped — run 'sudo -E bash deploy/bootstrap-{aws,gcp}.sh' first."
fi

if ! systemctl list-unit-files "${SERVICE}" >/dev/null 2>&1; then
    die "${SERVICE} not registered with systemd. \
Run the full bootstrap to install the unit, then come back here."
fi

# Stop the unit before reinstalling — pip writes into the same site-packages
# the running process imports from, and overwriting an in-use shared library
# is a known source of weird half-state errors. Stop is idempotent.
log "stopping ${SERVICE}"
systemctl stop "${SERVICE}"

# ─── pip install --force-reinstall ───────────────────────────────────────────

log "reinstalling xrlenv from ${REPO} into ${INSTALL_ROOT}/.venv (non-editable)"
if ! "${VENV_PIP}" install --no-deps --force-reinstall "${REPO}"; then
    warn "pip install failed; service is still stopped — fix the error and re-run"
    exit 3
fi

# Sanity-check that the new install actually imports — same `cd /` trick the
# bootstrap uses so the source tree under XRLENV_REPO doesn't shadow the
# venv copy via Python's CWD-on-sys.path.
if ! ( cd / && "${VENV_PYTHON}" -c "import xrlenv.node, xrlenv.node.cli" 2>/dev/null ); then
    die "post-install import check failed — xrlenv.node is unreachable from \
${INSTALL_ROOT}/.venv. Inspect with: \
  cd / && ${VENV_PYTHON} -c 'import xrlenv.node'"
fi

# Optional: surface the installed version + a fingerprint so the operator
# can confirm the bits actually changed. ``pip show`` is cheap.
"${VENV_PIP}" show xrlenv | awk '/^(Name|Version|Location):/ {print "[xrlenv-refresh] " $0}'

# ─── Additive config refresh ─────────────────────────────────────────────────
#
# Pull operator-set values out of the existing /etc/xrlenv/node.env
# (XRLENV_CONTROL_PLANE / XRLENV_NODE_ID were chosen by the operator at
# first bootstrap and shouldn't change on a refresh) so we can reuse
# bootstrap-common.sh's helpers without asking the operator to re-export
# them. Then re-run ensure_directories (idempotent mkdir + chown — adds
# any new subdirs the latest source expects) and install_systemd_unit
# (rewrites node.env from the heredoc, which picks up any new
# ``XRLENV_*`` env vars the latest source needs the daemon to see).
#
# install_systemd_unit also restarts the service via systemctl, which
# is fine — the refresh's own start step below is a no-op against an
# already-active unit.
NODE_ENV="${ETC_DIR}/node.env"
if [[ ! -f "${NODE_ENV}" ]]; then
    die "no ${NODE_ENV} on this host. Run the full bootstrap first
(sudo -E bash deploy/bootstrap-{aws,gcp}.sh with XRLENV_CONTROL_PLANE +
XRLENV_NODE_ID set) — refresh.sh assumes the bootstrap-managed config
already exists."
fi
# Issue #18 (Ask #2, audit M1 round 2): capture an operator-supplied
# XRLENV_BUILD_SHA *before* sourcing node.env. The existing node.env
# carries the PREVIOUS install's XRLENV_BUILD_SHA, and ``set -a;
# source`` below would import it into the shell — clobbering the
# value we resolve for THIS refresh. Saving the operator's env value
# here lets the resolution block further down tell "operator passed
# it" apart from "stale node.env had it".
_operator_build_sha="${XRLENV_BUILD_SHA:-}"
# Source existing config into the current shell so ensure_directories
# / install_systemd_unit see the right values. ``set -a`` exports
# every assignment; ``set +a`` restores normal scoping after.
set -a
# shellcheck disable=SC1090
source "${NODE_ENV}"
set +a
# Sanity: bootstrap-common.sh's heredoc requires both. If either is
# missing from the existing node.env (corrupted / hand-edited), bail
# loudly rather than rewrite with empty values.
: "${XRLENV_CONTROL_PLANE:?node.env missing XRLENV_CONTROL_PLANE; restore from backup or re-run the bootstrap}"
: "${XRLENV_NODE_ID:?node.env missing XRLENV_NODE_ID; restore from backup or re-run the bootstrap}"

# Resolve the build SHA for THIS refresh — AFTER the node.env source
# so the previous install's value (now in the env) can't survive.
# An operator-supplied SHA wins; otherwise derive from REPO, the
# checkout just reinstalled. ``install_systemd_unit`` writes the
# result into node.env.
if [[ -n "${_operator_build_sha}" ]]; then
    XRLENV_BUILD_SHA="${_operator_build_sha}"
else
    unset XRLENV_BUILD_SHA  # drop the stale value sourced from node.env
    XRLENV_BUILD_SHA="$(resolve_build_sha "${REPO}")"
fi
export XRLENV_BUILD_SHA
log "stamping node-agent build SHA: ${XRLENV_BUILD_SHA}"

log "refreshing /var/cache/xrlenv subdirs + ${NODE_ENV} from latest source"
ensure_directories
# Docker Hub auth refresh: when DOCKERHUB_USER + DOCKERHUB_TOKEN are
# present in the caller's env (typical operator invocation:
# ``sudo DOCKERHUB_USER=... DOCKERHUB_TOKEN=... bash deploy/refresh.sh``),
# rewrite the runtime user's ``~/.docker/config.json`` BEFORE
# install_systemd_unit restarts the daemon. docker-py's
# ``APIClient.__init__`` reads the auth file once at construction and
# caches it; rotating the PAT only takes effect on the next daemon
# start. Same ordering invariant as ``bootstrap_xrlenv`` — auth
# before unit-restart, locked in by the regression test at
# ``tests/unit/deploy/test_bootstrap_dockerhub_auth.py``.
#
# Skips silently when the env vars aren't passed — preserves the
# operator's existing config (the bash function early-returns when
# either var is missing AND a prior config.json is on disk). New
# operators get the loud advisory at end-of-refresh, same as
# bootstrap.
install_dockerhub_auth_credentials
install_systemd_unit

# ─── systemctl restart + smoke check ─────────────────────────────────────────
#
# install_systemd_unit above already restarted the service (the
# heredoc rewrite of node.env requires daemon-reload + restart for
# the daemon to see new env vars). The block below is the smoke
# check — verifies the unit landed on ``active`` and didn't crash
# loop.

log "starting ${SERVICE}"
if ! systemctl start "${SERVICE}"; then
    warn "${SERVICE} failed to start; check 'journalctl -u ${SERVICE} -n 50'"
    exit 4
fi

# Wait a moment for the unit to leave 'activating' and either land on
# 'active' or trip into the restart loop.
sleep 1
state="$(systemctl is-active "${SERVICE}" 2>/dev/null || true)"
if [[ "${state}" != "active" ]]; then
    warn "${SERVICE} state=${state}; tailing logs:"
    journalctl -u "${SERVICE}" -n 20 --no-pager || true
    exit 4
fi

ok "refresh complete: ${SERVICE} is active. Tail with 'journalctl -u ${SERVICE} -f'"
# Loud advisory if no Docker Hub auth was wired (same as bootstrap).
# Self-short-circuits when a config.json is already on disk, so
# operators who set it once don't get spammed every refresh.
warn_if_no_dockerhub_auth
