#!/usr/bin/env bash
# bootstrap-common.sh — shared install logic for xrlenv node agents.
#
# Sourced by deploy/bootstrap-{gcp,aws}.sh; not run directly. Idempotent;
# safe to re-run on a node that's already partially set up.
#
# Inputs (must be set before sourcing):
#   XRLENV_CONTROL_PLANE  e.g. "control.example.com:50051"
#   XRLENV_NODE_ID        stable identifier for this node
#   XRLENV_VERSION        wheel/version pin (default: "main")
#
# Side effects:
#   - installs Docker (if missing)
#   - creates /etc/xrlenv/, /var/lib/xrlenv/, /var/cache/xrlenv/
#   - installs the xrlenv wheel into /opt/xrlenv/.venv
#   - drops the systemd unit at /etc/systemd/system/xrlenv-node.service
#   - enables + starts the service

set -euo pipefail

# Re-source the preflight helpers (color codes + die/log/ok/warn). The
# parent script (bootstrap-{aws,gcp}.sh) sources _preflight.sh first;
# the sentinel in _preflight.sh makes this re-source a no-op there.
# Direct callers of bootstrap-common.sh (rare) get the helpers here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_preflight.sh
source "${SCRIPT_DIR}/_preflight.sh"

# IMPORTANT: this file is sourced by both ``bootstrap_xrlenv()`` (full
# install) and ``deploy/refresh.sh`` (source-only update path). The
# refresh path needs to call ``ensure_directories`` /
# ``install_systemd_unit`` without firing the bootstrap-only env
# checks, so all ``require_env`` calls live inside
# ``validate_required_env_for_bootstrap`` rather than at module level.
# Sourcing this file is side-effect-free; the entry point is
# ``bootstrap_xrlenv``.

: "${XRLENV_VERSION:=main}"

INSTALL_ROOT="/opt/xrlenv"
RUNTIME_USER="${XRLENV_USER:-xrlenv}"
ETC_DIR="/etc/xrlenv"
SCRATCH_DIR="/var/lib/xrlenv"
CACHE_DIR="/var/cache/xrlenv"
SYSTEMD_UNIT="/etc/systemd/system/xrlenv-node.service"

ensure_user() {
    if ! id -u "${RUNTIME_USER}" >/dev/null 2>&1; then
        log "creating system user ${RUNTIME_USER}"
        useradd --system --shell /usr/sbin/nologin --home-dir "${INSTALL_ROOT}" "${RUNTIME_USER}"
    fi
    # Docker group membership is required for the agent to dial the docker socket.
    usermod -aG docker "${RUNTIME_USER}" || log "WARN: docker group missing; will retry after Docker install"
}

# Add the operator's interactive user (the one who invoked ``sudo
# bootstrap-...sh``) to the docker group, so they can run the
# per-VM build-task-images.sh script without sudo. The bootstrap
# already gives the system ``xrlenv`` user docker access for the
# daemon; this closes the same loop for the human at the keyboard.
#
# - Skipped when ``$SUDO_USER`` is unset (running as root directly,
#   no operator user to identify).
# - Skipped when ``XRLENV_SKIP_OPERATOR_DOCKER_GROUP=1`` (opt-out
#   for hardened-security or multi-operator setups where the
#   operator prefers explicit control).
# - The new GID does NOT take effect in the operator's existing
#   shell; they need to re-login or run ``newgrp docker``. We log
#   the reminder.
ensure_operator_docker_group() {
    if [[ "${XRLENV_SKIP_OPERATOR_DOCKER_GROUP:-0}" == "1" ]]; then
        log "skipping operator docker-group add (XRLENV_SKIP_OPERATOR_DOCKER_GROUP=1)"
        return 0
    fi
    local op_user="${SUDO_USER:-}"
    if [[ -z "$op_user" || "$op_user" == "root" ]]; then
        log "no operator user to add to docker group (SUDO_USER unset or root)"
        return 0
    fi
    if ! id -u "$op_user" >/dev/null 2>&1; then
        log "WARN: SUDO_USER=$op_user does not exist; skipping docker-group add"
        return 0
    fi
    if id -nG "$op_user" 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
        log "operator $op_user already in docker group"
        return 0
    fi
    if usermod -aG docker "$op_user" 2>/dev/null; then
        log "added operator $op_user to docker group"
        log "  → re-login (or run ``newgrp docker``) for the new GID to apply"
    else
        log "WARN: usermod -aG docker $op_user failed; the operator can run"
        log "      ``sudo usermod -aG docker $op_user && newgrp docker`` manually"
    fi
}

ensure_directories() {
    mkdir -p "${INSTALL_ROOT}" "${ETC_DIR}" "${SCRATCH_DIR}" "${CACHE_DIR}"
    # Shared harbor task cache. The systemd unit reads this path
    # via XRLENV_BENCHMARK_CACHE in node.env; the per-VM image-build
    # script (and operator-driven ``populate-harbor-cache.sh``) also
    # write here. World-readable so an interactive operator user can
    # ``ls`` it without sudo for diagnostics; group-writable for the
    # ``xrlenv`` system user the daemon runs as.
    mkdir -p "${CACHE_DIR}/harbor/tasks"
    # Build-context cache for git/tarball-source plans. The
    # GitSourceBuilder clones repos into this tree and reuses
    # them across builds (LRU evict at 5 GB total cap). Sits under
    # ${CACHE_DIR} (= /var/cache/xrlenv) per FHS — cache state is
    # regeneratable, not /var/lib durable state. The systemd unit
    # reads the path via XRLENV_BUILD_CONTEXT_CACHE in node.env;
    # without the env var the daemon would mkdir under the
    # xrlenv user's $HOME, which the systemd unit mounts read-only
    # via ProtectHome=read-only — that's why an explicit writable
    # path is required.
    mkdir -p "${CACHE_DIR}/build-context-cache"
    chown -R "${RUNTIME_USER}":"${RUNTIME_USER}" "${INSTALL_ROOT}" "${SCRATCH_DIR}" "${CACHE_DIR}"
    chmod 0775 "${CACHE_DIR}/harbor" "${CACHE_DIR}/harbor/tasks"
}

_python_meets_requirement() {
    # _python_meets_requirement <python-bin> — exits 0 iff the binary
    # exists and reports sys.version_info >= (3, 12). Used to validate
    # both freshly-installed interpreters and pre-existing venvs.
    local bin="$1"
    [[ -x "$(command -v "${bin}" 2>/dev/null || echo "${bin}")" ]] || return 1
    "${bin}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null
}

_install_python_via_uv() {
    # Last-resort Python 3.12 installer: download `uv` (Astral's static
    # binary) and use `uv python install 3.12`, which fetches a
    # python-build-standalone distribution that runs on any glibc Linux.
    # Used when neither the distro repos nor an operator-pinned
    # XRLENV_PYTHON yields a 3.12+ interpreter — e.g. Ubuntu 22.04
    # where add-apt-repository:ppa:deadsnakes is fragile, or Debian
    # where deadsnakes is not applicable at all.
    local install_dir="${INSTALL_ROOT}/python"
    mkdir -p "${install_dir}"
    if ! command -v uv >/dev/null 2>&1; then
        local arch uv_target tmpdir
        arch="$(uname -m)"
        case "${arch}" in
            x86_64)  uv_target="x86_64-unknown-linux-gnu" ;;
            aarch64) uv_target="aarch64-unknown-linux-gnu" ;;
            *)
                die "uv prebuilt binaries do not cover arch '${arch}'. \
Install Python 3.12 manually (e.g. compile from source under /opt) and \
re-run with XRLENV_PYTHON=/path/to/python3.12 set in the environment."
                ;;
        esac
        log "downloading uv (Astral) for ${uv_target}"
        tmpdir="$(mktemp -d)"
        if ! curl -fsSL "https://github.com/astral-sh/uv/releases/latest/download/uv-${uv_target}.tar.gz" \
                | tar -xzC "${tmpdir}"; then
            die "failed to download uv from github.com. \
Check the VM's outbound internet (proxy, egress firewall, DNS), \
or pre-install python3.12 on this VM and re-run with XRLENV_PYTHON set."
        fi
        install -m 0755 "${tmpdir}/uv-${uv_target}/uv" /usr/local/bin/uv
        rm -rf "${tmpdir}"
        ok "uv installed at /usr/local/bin/uv"
    fi
    log "installing portable Python 3.12 via uv into ${install_dir}"
    UV_PYTHON_INSTALL_DIR="${install_dir}" uv python install 3.12
    local py_path
    py_path="$(find "${install_dir}" -path '*/bin/python3.12' -type f -executable 2>/dev/null | head -n1)"
    if [[ -z "${py_path}" || ! -x "${py_path}" ]]; then
        die "uv reported success but no python3.12 binary found under ${install_dir}. \
Inspect that directory, then re-run with XRLENV_PYTHON pointing at the binary."
    fi
    export XRLENV_PYTHON="${py_path}"
    ok "uv-managed Python ready at ${XRLENV_PYTHON}"
}

ensure_python_312() {
    # Goal: leave XRLENV_PYTHON pointing at a Python >= 3.12. Order:
    #   1. Honor an operator-pinned XRLENV_PYTHON if it satisfies >=3.12.
    #   2. Look for python3.{14,13,12} on PATH (already-native).
    #   3. Try the distro's package manager (dnf for AL2023/Fedora/RHEL,
    #      apt for Ubuntu 24.04+ which carries python3.12 natively).
    #   4. Fall back to uv-managed python-build-standalone.
    if [[ -n "${XRLENV_PYTHON:-}" ]] && _python_meets_requirement "${XRLENV_PYTHON}"; then
        log "using operator-pinned XRLENV_PYTHON=${XRLENV_PYTHON}"
        return 0
    fi
    for candidate in python3.14 python3.13 python3.12; do
        if _python_meets_requirement "${candidate}"; then
            export XRLENV_PYTHON="${candidate}"
            log "found native ${candidate} on PATH"
            return 0
        fi
    done
    log "python3.12+ not on PATH; attempting native install via the distro"
    . /etc/os-release
    case "${ID:-}" in
        amzn|rhel|fedora)
            if dnf install -y python3.12 python3.12-pip 2>/dev/null \
               && _python_meets_requirement python3.12; then
                export XRLENV_PYTHON="python3.12"
                ok "installed python3.12 via dnf"
                return 0
            fi
            ;;
        ubuntu|debian)
            if apt-get install -y --no-install-recommends python3.12 python3.12-venv 2>/dev/null \
               && _python_meets_requirement python3.12; then
                export XRLENV_PYTHON="python3.12"
                ok "installed python3.12 via apt"
                return 0
            fi
            warn "python3.12 not available from this distro's apt repos (expected on Ubuntu 22.04 and Debian 12)"
            ;;
        *)
            warn "unrecognised distro '${ID:-unknown}'; skipping native install attempt"
            ;;
    esac
    _install_python_via_uv
}

install_python_venv() {
    # xrlenv requires Python >= 3.12 (see pyproject.toml). ensure_python_312
    # runs first to guarantee XRLENV_PYTHON points at a satisfying interpreter
    # — whether installed by the distro or downloaded via uv. We then build
    # the venv with that interpreter and verify the version a second time
    # in case the operator passed a stale XRLENV_PYTHON pointing at an
    # already-built /opt/xrlenv/.venv from a previous run on python3.9.
    ensure_python_312
    local py_bin="${XRLENV_PYTHON}"
    local py_version
    py_version=$("${py_bin}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    log "using ${py_bin} (Python ${py_version}) to build the venv"

    if [[ ! -x "${INSTALL_ROOT}/.venv/bin/python" ]]; then
        log "creating Python venv at ${INSTALL_ROOT}/.venv"
        "${py_bin}" -m venv "${INSTALL_ROOT}/.venv"
    else
        # Existing venv: verify it was built with a 3.12+ interpreter,
        # otherwise the upcoming pip install will hit the same resolver
        # error the user just reported. Fail loud with the fix.
        local venv_version
        venv_version=$("${INSTALL_ROOT}/.venv/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
        if ! _python_meets_requirement "${INSTALL_ROOT}/.venv/bin/python"; then
            die "Existing venv at ${INSTALL_ROOT}/.venv was built with Python ${venv_version}, but xrlenv requires >= 3.12. \
Remove the stale venv and re-run: 'sudo rm -rf ${INSTALL_ROOT}/.venv && sudo -E bash deploy/bootstrap-...sh'."
        fi
    fi
    "${INSTALL_ROOT}/.venv/bin/pip" install --upgrade pip
    # Three install paths, in priority order:
    #   1. XRLENV_WHEEL  — explicit wheel path (production bootstraps)
    #   2. XRLENV_REPO   — checkout dir, installed *non-editable* (see note)
    #   3. PyPI fallback — `pip install xrlenv==${XRLENV_VERSION}`
    # Whichever path runs, `xrlenv-node` must end up on the venv's PATH
    # AND `xrlenv.node` must be importable by the runtime user — the
    # systemd unit drops to User=xrlenv with ProtectHome=read-only, so
    # an editable install whose .pth points at /home/<sudo-user>/xrlenv
    # is unreachable (mode-700 home dir blocks traversal). Non-editable
    # install copies sources into /opt/xrlenv/.venv/lib/.../site-packages
    # which the xrlenv user owns, sidestepping the issue. The cost: the
    # operator has to re-run the bootstrap (idempotent + fast) to pick
    # up source edits — fine for an acceptance smoke, where node
    # code rarely changes.
    # Issue #18 (Ask #2, audit M1): stamp the build SHA from the
    # install source that actually ran — only the checkout path can
    # supply a real SHA; wheel / PyPI fall back to a caller-supplied
    # value or "unknown". Exported so ``install_systemd_unit`` (called
    # later by ``bootstrap_xrlenv``) writes it into node.env.
    if [[ -n "${XRLENV_WHEEL:-}" ]]; then
        log "installing xrlenv from wheel: ${XRLENV_WHEEL}"
        "${INSTALL_ROOT}/.venv/bin/pip" install "${XRLENV_WHEEL}"
        XRLENV_BUILD_SHA="$(resolve_build_sha "")"
    elif [[ -n "${XRLENV_REPO:-}" && -f "${XRLENV_REPO}/pyproject.toml" ]]; then
        log "installing xrlenv from checkout (non-editable): ${XRLENV_REPO}"
        "${INSTALL_ROOT}/.venv/bin/pip" install "${XRLENV_REPO}"
        XRLENV_BUILD_SHA="$(resolve_build_sha "${XRLENV_REPO}")"
    else
        log "installing xrlenv from PyPI tag: ${XRLENV_VERSION}"
        if ! "${INSTALL_ROOT}/.venv/bin/pip" install "xrlenv==${XRLENV_VERSION}"; then
            warn "PyPI install failed (expected during pre-release)."
            warn "Set XRLENV_WHEEL=/path/to/xrlenv-X.Y.Z-py3-none-any.whl, or"
            warn "set XRLENV_REPO=/path/to/checkout (containing pyproject.toml) and re-run."
        fi
        XRLENV_BUILD_SHA="$(resolve_build_sha "")"
    fi
    export XRLENV_BUILD_SHA
    if [[ ! -x "${INSTALL_ROOT}/.venv/bin/xrlenv-node" ]]; then
        die "xrlenv-node not on the venv PATH after install. \
The systemd unit will fail to start. Choose one of: \
(a) build a wheel locally and re-run with XRLENV_WHEEL=<path>; \
(b) re-run with XRLENV_REPO=<checkout dir>."
    fi
    # Sanity-check that xrlenv.node is actually importable. The xrlenv-node
    # CLI script's first line is `from xrlenv.node.cli import main`; if
    # that fails, the systemd unit goes into a restart loop with
    # `ModuleNotFoundError`. Catch it here with the actual traceback.
    #
    # We `cd /` first so Python's CWD-on-sys.path injection doesn't shadow
    # the venv install with the source tree under XRLENV_REPO — otherwise
    # this check validates the wrong copy, and a broken venv install can
    # silently pass while the systemd unit (with WorkingDirectory=/opt/xrlenv)
    # hits the actual breakage at startup.
    local import_check_log
    import_check_log="$(mktemp)"
    if ! ( cd / && "${INSTALL_ROOT}/.venv/bin/python" \
            -c "import xrlenv.node, xrlenv.node.cli" 2>"${import_check_log}" ); then
        local trace
        trace="$(cat "${import_check_log}")"
        rm -f "${import_check_log}"
        die "xrlenv.node not importable from ${INSTALL_ROOT}/.venv. \
Actual error from the venv-installed copy:
${trace}
If this looks like a circular import or a missing dep, fix it on the \
source side (the systemd unit will hit the same error at startup). \
If the message is 'No module named xrlenv.node', a stray editable .pth \
may be present — wipe with 'sudo rm -rf ${INSTALL_ROOT}/.venv' and re-run."
    fi
    rm -f "${import_check_log}"
    ok "xrlenv.node imports cleanly from the venv"
}

# Issue #18 (Ask #2, audit M1): resolve the build SHA of the xrlenv
# package source ACTUALLY being installed — not the checkout that
# happens to contain these deploy scripts (the two can differ:
# ``XRLENV_REPO=/elsewhere``, a wheel install, a PyPI install, or
# ``refresh.sh /path/to/other/checkout``). Stamping the wrong SHA
# would make NodeHello.agent_version falsely match / falsely skew,
# defeating the stale-node detector.
#
# Resolution order:
#   1. A caller-supplied ``XRLENV_BUILD_SHA`` always wins — the CI
#      that built a wheel knows its SHA even though the wheel has no
#      ``.git``.
#   2. ``git rev-parse`` against the passed install-source checkout.
#   3. ``unknown`` — an install whose artifact genuinely can't
#      identify itself (bare wheel / PyPI with no caller SHA).
resolve_build_sha() {
    if [[ -n "${XRLENV_BUILD_SHA:-}" ]]; then
        printf '%s\n' "${XRLENV_BUILD_SHA}"
        return 0
    fi
    local src="${1:-}" sha=""
    if [[ -n "${src}" ]]; then
        sha="$(git -C "${src}" rev-parse --short=12 HEAD 2>/dev/null)" || sha=""
    fi
    printf '%s\n' "${sha:-unknown}"
}

install_systemd_unit() {
    log "writing systemd unit to ${SYSTEMD_UNIT}"
    install -m 0644 "$(dirname "${BASH_SOURCE[0]}")/systemd/xrlenv-node.service" "${SYSTEMD_UNIT}"
    cat >"${ETC_DIR}/node.env" <<EOF
XRLENV_CONTROL_PLANE=${XRLENV_CONTROL_PLANE}
XRLENV_NODE_ID=${XRLENV_NODE_ID}
# Build SHA of the xrlenv package source this node-agent was
# installed from (issue #18 Ask #2). Surfaced via xrlenv.buildinfo
# into NodeHello so the control plane WARNs on version skew.
# Resolved by the calling entrypoint from the actual install source;
# rewritten on every bootstrap / refresh.sh run.
XRLENV_BUILD_SHA=${XRLENV_BUILD_SHA:-unknown}
# Issue #18 — how many distinct images this node-agent pulls
# concurrently (xrlenv.node.image_cache ImageCacheConfig.pull_
# concurrency). The xrlenv library default is 2, tuned for
# image-reuse-heavy RL training where cold pulls are rare. We stamp
# 6 here because cold-pull-heavy benchmark workloads (a unique
# multi-GB image per task — SWE-bench Pro etc.) leave the network
# link idle between pulls at concurrency 2. Lower it back toward 2
# for reuse-heavy training; raise it cautiously — many concurrent
# pulls also mean many concurrent registry-auth requests, and that
# endpoint contends under load. Edit + restart xrlenv-node to apply;
# blank / non-positive falls back to the library default of 2.
# Adaptive (AIMD) image-pull concurrency. A single node-local limiter
# moves between a floor and a ceiling based on node load: busy with live
# rollouts → multiplicative-decrease toward the floor (so cold pulls
# never slow time-sensitive agents); idle (e.g. ``xrlenv build apply``)
# → additive-increase toward the ceiling (saturate the registry/FSx
# pipe). Edit + restart xrlenv-node to apply; blank / non-positive falls
# back to the library defaults (floor 2 / ceiling 64 / initial 16).
XRLENV_PULL_CONCURRENCY=${XRLENV_PULL_CONCURRENCY:-2}            # AIMD floor
XRLENV_PULL_CONCURRENCY_CEILING=${XRLENV_PULL_CONCURRENCY_CEILING:-64}
XRLENV_PULL_CONCURRENCY_INITIAL=${XRLENV_PULL_CONCURRENCY_INITIAL:-16}
# Shared harbor task cache — same path the build script and the
# operator's ``populate-harbor-cache.sh`` write to. Without this the
# daemon falls back to ``\$HOME/.cache/harbor/tasks`` (= ``\$HOME``
# of the ``xrlenv`` user, /opt/xrlenv) which the operator typically
# never populates → empty cache → every rollout fails resolution.
XRLENV_BENCHMARK_CACHE=${CACHE_DIR}/harbor/tasks
# Build-context cache for git/tarball-source plans. The
# ``GitSourceBuilder`` clones repos here and reuses them across
# builds (LRU cap, default 5 GB). Without this env var the daemon
# would mkdir under \$HOME/.xrlenv, which the systemd unit's
# ProtectHome=read-only blocks → every git build fails with
# OSError: Read-only file system. Sits under /var/cache/xrlenv
# per FHS — regeneratable cache, safe to ``rm -rf`` between runs;
# next build re-clones from upstream.
XRLENV_BUILD_CONTEXT_CACHE=${CACHE_DIR}/build-context-cache
EOF
    chown root:"${RUNTIME_USER}" "${ETC_DIR}/node.env"
    chmod 0640 "${ETC_DIR}/node.env"
    install_node_token_dropin
    systemctl daemon-reload
    systemctl enable xrlenv-node.service
    systemctl restart xrlenv-node.service
    ok "xrlenv-node service started; tail logs via 'journalctl -u xrlenv-node -f'"
}

# Wire ``XRLENV_NODE_TOKEN`` (issued by the operator on the control
# plane via ``xrlenv tokens issue node``) into a systemd drop-in so
# the daemon authenticates against the gRPC interceptor on first
# connect. Skipped when unset — operators running unauthenticated
# smokes can leave it out.
#
# Drop-in (vs editing /etc/xrlenv/node.env directly): keeps secrets
# out of the world-readable EnvironmentFile and survives bootstrap
# re-runs without clobbering the token. Mode 0600, root-owned.
install_node_token_dropin() {
    local dropin_dir="${SYSTEMD_UNIT}.d"
    local dropin="${dropin_dir}/10-token.conf"
    if [[ -z "${XRLENV_NODE_TOKEN:-}" ]]; then
        if [[ -f "${dropin}" ]]; then
            log "preserving existing node-token drop-in at ${dropin}"
        else
            log "no XRLENV_NODE_TOKEN set; skipping token drop-in (set it for authenticated control planes)"
        fi
        return 0
    fi
    mkdir -p "${dropin_dir}"
    cat >"${dropin}" <<EOF
[Service]
Environment="XRLENV_NODE_TOKEN=${XRLENV_NODE_TOKEN}"
EOF
    chown root:root "${dropin}"
    chmod 0600 "${dropin}"
    log "wrote node-token drop-in to ${dropin} (mode 0600)"
}

# Write a Docker Hub auth config under the runtime user's home so
# ``docker pull`` calls initiated by ``xrlenv-node`` are authenticated
# and don't burn the unauth ~100-pulls-per-6h-per-IP cap. End users
# submitting jobs to the control plane never touch Docker Hub
# directly; this is the only step that decides whether their cold
# acquires (and large ``xrlenv build apply`` sweeps) are auth'd.
#
# Direct config.json write (vs ``docker login``) — no TTY needed,
# idempotent on re-run, byte-identical to what ``docker login``
# produces. Mode 0600, owned by the runtime user (= ``xrlenv``).
#
# Skipped silently when DOCKERHUB_USER + DOCKERHUB_TOKEN aren't set;
# ``warn_if_no_dockerhub_auth`` prints a loud one-time WARN at
# end-of-bootstrap so the operator sees the gap immediately rather
# than 30 min later when the first large pull stalls.
install_dockerhub_auth_credentials() {
    local docker_dir="${INSTALL_ROOT}/.docker"
    local config_path="${docker_dir}/config.json"

    if [[ -z "${DOCKERHUB_USER:-}" || -z "${DOCKERHUB_TOKEN:-}" ]]; then
        if [[ -f "${config_path}" ]]; then
            log "preserving existing Docker Hub auth at ${config_path}"
        fi
        return 0
    fi

    mkdir -p "${docker_dir}"
    local auth_b64
    auth_b64=$(printf '%s' "${DOCKERHUB_USER}:${DOCKERHUB_TOKEN}" \
        | base64 | tr -d '\n')
    cat >"${config_path}" <<EOF
{
  "auths": {
    "https://index.docker.io/v1/": {
      "auth": "${auth_b64}"
    }
  }
}
EOF
    chown -R "${RUNTIME_USER}":"${RUNTIME_USER}" "${docker_dir}"
    chmod 0700 "${docker_dir}"
    chmod 0600 "${config_path}"
    log "wrote Docker Hub auth to ${config_path} (owner ${RUNTIME_USER}, mode 0600)"
}

# Print a loud end-of-bootstrap warning if no Docker Hub creds were
# wired up and no prior config.json survives. The operator's typical
# next step is to apply a build plan; without auth, the daemon will
# rate-limit around image #100 and an apparently-reasonable plan
# rejects with InsufficientCapacity-shaped errors. Catching this at
# bootstrap time saves the 30-min debug round-trip.
warn_if_no_dockerhub_auth() {
    if [[ -n "${DOCKERHUB_USER:-}" && -n "${DOCKERHUB_TOKEN:-}" ]]; then
        return 0
    fi
    if [[ -f "${INSTALL_ROOT}/.docker/config.json" ]]; then
        return 0
    fi
    cat >&2 <<EOF

============================================================
WARNING: no Docker Hub auth configured on this node.
The docker daemon will rate-limit at ~100 image pulls / 6h
per source IP. For large sweeps (e.g. 500-instance SWE-bench
Verified) end users submitting jobs will hit
InsufficientCapacity-shaped failures partway through.

Fix at any time, one of:
  # (preferred) re-run this bootstrap with creds set:
  export DOCKERHUB_USER=<your-handle>
  export DOCKERHUB_TOKEN=<your-PAT>
  sudo -E bash deploy/bootstrap-{gcp,aws}.sh

  # or, log in as the runtime user on this node directly:
  sudo -u ${RUNTIME_USER} docker login

A Docker Hub Personal Access Token (PAT) lives at
https://hub.docker.com/settings/security — Business / Pro /
Team tiers carry a much higher per-account pull cap.
============================================================
EOF
}

validate_required_env_for_bootstrap() {
    # Operator-set knobs the bootstrap can't infer.
    require_env XRLENV_CONTROL_PLANE \
        "Pass control-plane host:port (e.g. XRLENV_CONTROL_PLANE=10.0.0.1:50051) before re-running."
    require_env XRLENV_NODE_ID \
        "Pass a stable node identifier (e.g. XRLENV_NODE_ID=aws-i-0123...) or rely on the bootstrap's metadata-service auto-detect."

    # Source pointer: bootstrap_xrlenv installs the xrlenv package
    # into /opt/xrlenv/.venv. Three valid sources, in priority
    # order: XRLENV_WHEEL > XRLENV_REPO > PyPI. xrlenv isn't on
    # PyPI today (pre-release), so falling through to PyPI yields
    # an unhelpful ``Invalid requirement: 'xrlenv==main'`` from
    # pip when XRLENV_VERSION is the default 'main' git ref. Catch
    # the common case up front with operator-friendly guidance,
    # rather than letting pip fail mid-install. The XRLENV_USE_PYPI
    # escape hatch lets a future operator with a real PyPI release
    # opt back into the PyPI path explicitly.
    if [[ -z "${XRLENV_WHEEL:-}" && -z "${XRLENV_REPO:-}" \
          && "${XRLENV_USE_PYPI:-0}" != "1" ]]; then
        die "no xrlenv source pointer set (xrlenv isn't on PyPI yet). Pass one of:
  XRLENV_REPO=/path/to/xrlenv/checkout    # most common — the source tree on this VM
  XRLENV_WHEEL=/path/to/xrlenv-X.Y.Z.whl  # for a pre-built wheel
  XRLENV_USE_PYPI=1                       # opt into PyPI install (fails today;
                                          # only useful once xrlenv ships a real release)

If the xrlenv repo is at \$(pwd) on this VM, the simplest fix is:
  export XRLENV_REPO=\$(pwd)
  sudo -E bash deploy/bootstrap-{gcp,aws}.sh"
    fi
}

bootstrap_xrlenv() {
    validate_required_env_for_bootstrap
    ensure_user
    ensure_directories
    install_python_venv
    # NOTE: install_dockerhub_auth_credentials MUST run before
    # install_systemd_unit. The latter calls ``systemctl restart
    # xrlenv-node``, which constructs ``DockerBackend`` via
    # ``docker.from_env()`` at process startup; docker-py's
    # ``APIClient.__init__`` reads ``~/.docker/config.json`` ONCE
    # at that moment and caches the auth dict on the client
    # instance. If we install the auth file after the daemon has
    # started, the running daemon process keeps its empty-auth
    # cache until the next ``systemctl restart``, and every pull
    # initiated through that client is unauthenticated. Audit
    # M1 (2026-05-12) — caught in the order-mattered review.
    install_dockerhub_auth_credentials
    install_systemd_unit
    ensure_operator_docker_group
    ok "bootstrap complete; node-id=${XRLENV_NODE_ID} dialing ${XRLENV_CONTROL_PLANE}"
    warn_if_no_dockerhub_auth
}
