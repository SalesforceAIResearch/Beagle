#!/usr/bin/env bash
# deploy_example.sh — one-command (re)deploy of the example xrlenv cluster.
#
# Does, in order:
#   1. restart the node job          (scancel + sbatch example_xrlenv_node.sh)
#   2. restart the control job       (scancel + sbatch example_xrlenv_control.sh)
#      + open the SSH tunnel to the control plane (admin :9080, metrics :9190)
#   3. configure + install the Sysbox node pool, IDEMPOTENTLY (skip any node
#      whose docker already advertises sysbox-runc)
#   4. enable the P6 CPU-isolation pool, IDEMPOTENTLY (flip docker to cgroupfs +
#      build a probe image so unpinned runc containers get complement-confined)
#
# Run from anywhere on the login node:  bash slurm_scripts/deploy_example.sh
# Idempotent: safe to re-run; a healthy cluster just gets bounced + re-tunnelled,
# and already-installed sysbox nodes are skipped (no needless docker restart).

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Example config — the ONLY block to edit
# ─────────────────────────────────────────────────────────────────────────────
NODE_JOB="example-xrlenv-nodes"
CONTROL_JOB="example-xrlenv-control"
NODE_SBATCH="slurm_scripts/example_xrlenv_node.sh"
CONTROL_SBATCH="slurm_scripts/example_xrlenv_control.sh"
CP_NODE="node-host"                       # control-plane host
CP_PORT=50051                                  # gRPC port (mirrors example_xrlenv_node.sh XRLENV_GRPC_PORT default)
STATE_DB="/opt/sagemaker/xrlenv/state.db"   # CP registry — CP-box-LOCAL NVMe (matches example_xrlenv_control.sh XRLENV_STATE_DB_PATH); liveness check reads it ON the CP box (XRLENV_VERIFY_CP_SSH_HOST)
TUNNEL_FWDS=(-L 9080:localhost:8080 -L 9190:localhost:9090)  # admin 8080, metrics 9090

# Sysbox node pool — hostnames to install Sysbox on AND declare in nodes.yaml
# (must be a subset of the node job's nodelist). Empty = no sysbox pool.
# Uncomment to enable DinD / systemd / netns tasks on the cluster.
# SYSBOX_POOL=()
SYSBOX_POOL=(node-host)
# Per-node cap on concurrently RUNNING sysbox containers — bounds sysbox-fs load
# so a create/exec storm can't wedge the FUSE daemon (2026-07-07 incident; see
# notes/design-per-node-runtime-concurrency-cap.md). Stamped onto every
# SYSBOX_POOL node at nodes.yaml generation; overflow queues in the admission
# queue. Empty = unlimited (unchanged). 8 leaves headroom below the ~22 that wedged.
SYSBOX_MAX_CONCURRENT=4
# Host bind paths to allow (policy.allowed_host_paths) — authorizes real host
# mounts on sysbox nodes (spec 19). EvoClaw's oracle bind-mounts its golden
# cache READ-ONLY into the container (EVOCLAW_GOLDEN_DIR, a subdir under this
# root), so the cache root must be allowlisted here. This is the SHARED cache
# (run_e2e_xrlenv.py default), not a personal home dir — prefix-matched, so this
# one entry covers every per-task mount under it. Empty = no extra host paths.
ALLOWED_HOST_PATHS=(/path/to/host-cache)
# P6 CPU-isolation pool — hostnames to make isolation-capable via
# scripts/enable_cpu_isolation.sh (flip docker to the cgroupfs driver + build a
# probe image, so the node's self-test passes and unpinned runc containers get
# complement-confined off the pinned cores — notes/cluster-resource-isolation-plan.md
# §8). NOT a nodes.yaml field: isolation_capable is self-test-discovered and
# advertised on NodeHello, so nothing changes in the roster. Enabling a node
# RESTARTS docker (bounces its containers); done idempotently below (a node
# already on the cgroupfs driver is skipped). P6 v1 is runc-only, so do NOT
# overlap with SYSBOX_POOL — flipping a sysbox node to cgroupfs can break its DinD
# workloads (a warning fires below). Empty = no P6 pool. Uncomment to enable.
CPU_ISOLATION_POOL=(node-host)
# CPU_ISOLATION_POOL=()
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10)
log() { printf '\n==> %s\n' "$*"; }
# Fail-closed pre/post-deploy safety gates (shared across clusters; unit-tested in
# tests/unit/deploy/test_deploy_gates.py). They use SSH_OPTS + XRLENV_DEPLOY_FORCE.
# shellcheck source=_deploy_gates.sh
source "${REPO_ROOT}/slurm_scripts/_deploy_gates.sh"

# Normalize the pool: accept either space- OR comma-separated entries (a bash
# array element like ``a,b`` is otherwise one literal token, which would silently
# fail to match a hostname). Flattens ``(a,b)`` and ``(a b)`` both to ``(a b)``.
_pool_norm=()
if [ "${#SYSBOX_POOL[@]}" -gt 0 ]; then
    for _e in "${SYSBOX_POOL[@]}"; do
        IFS=', ' read -ra _parts <<< "${_e}"
        _pool_norm+=("${_parts[@]}")
    done
fi
SYSBOX_POOL=("${_pool_norm[@]}")

_iso_norm=()
if [ "${#CPU_ISOLATION_POOL[@]}" -gt 0 ]; then
    for _e in "${CPU_ISOLATION_POOL[@]}"; do
        IFS=', ' read -ra _parts <<< "${_e}"
        _iso_norm+=("${_parts[@]}")
    done
fi
CPU_ISOLATION_POOL=("${_iso_norm[@]}")

# ── 1) restart the node job ──────────────────────────────────────────────────
log "restarting nodes: ${NODE_JOB}"
scancel --name="${NODE_JOB}" 2>/dev/null || true
sleep 2
sbatch "${NODE_SBATCH}"

log "waiting for ${NODE_JOB} to be RUNNING (nodes reachable for sysbox install)..."
deploy_wait_node_running "${NODE_JOB}"   # fail-closed on timeout (XRLENV_DEPLOY_FORCE=1 overrides)
squeue --name="${NODE_JOB}" 2>/dev/null || true

# ── 2) restart the control job (declare sysbox pool via env) + open tunnel ────
log "restarting control: ${CONTROL_JOB} (sysbox pool: ${SYSBOX_POOL[*]:-none})"
# The control job's nodes-from-slurm reads XRLENV_SYSBOX_POOL to mark
# `sysbox: true` + permit sysbox-runc in policy.allowed_runtimes at generation
# time (empty = no-op). sbatch --export=ALL propagates it into the job.
export XRLENV_SYSBOX_POOL="${SYSBOX_POOL[*]:-}"
# …and XRLENV_SYSBOX_MAX_CONCURRENT stamps the per-node sysbox concurrency cap
# on those pool nodes at generation (empty = unlimited). Durable across regen: a
# per-node value already in nodes.yaml is preserved and wins over this default.
export XRLENV_SYSBOX_MAX_CONCURRENT="${SYSBOX_MAX_CONCURRENT:-}"
# …and XRLENV_ALLOWED_HOST_PATHS (newline-separated) is merged into
# policy.allowed_host_paths at generation, so the EvoClaw golden-mount allowlist
# is an env knob here instead of a personal path baked into the committed roster.
export XRLENV_ALLOWED_HOST_PATHS="$(printf '%s\n' "${ALLOWED_HOST_PATHS[@]:-}")"
scancel --name="${CONTROL_JOB}" 2>/dev/null || true
# Wait for the OLD control plane to FULLY exit before starting a new one — two
# `xrlenv up` on one state.db corrupt the SQLite -shm / race the journal.
log "waiting for ${CONTROL_JOB} to fully exit (avoid overlapping state.db open)..."
deploy_wait_control_gone "${CONTROL_JOB}" "${CP_NODE}"   # fail-closed (XRLENV_DEPLOY_FORCE=1 overrides)
# state.db lives on the CP box's local NVMe (/opt/sagemaker, root-owned). Pre-create its
# parent dir user-owned so the control job (runs as us) can write it — passwordless sudo on
# the box. Idempotent; survives across restarts on the local NVMe.
log "ensuring local state.db dir on ${CP_NODE}: $(dirname "${STATE_DB}")"
ssh "${SSH_OPTS[@]}" "${CP_NODE}" \
    "sudo mkdir -p '$(dirname "${STATE_DB}")' && sudo chown \"\$(id -un):\$(id -gn)\" '$(dirname "${STATE_DB}")'" \
    || { [ "${XRLENV_DEPLOY_FORCE:-}" = 1 ] || { echo "ERROR: could not create local state.db dir on ${CP_NODE}" >&2; exit 1; }; }
# Propagate STATE_DB into the control job as XRLENV_STATE_DB_PATH (single source of truth for
# where the CP writes state.db) — via --export so it can't drift from the liveness check +
# dir-precreate above. `ALL,VAR=val` keeps the other propagated env (sysbox pool) too, and
# avoids exporting it into THIS script's env (whose xrlenv calls run on the login box).
sbatch --export="ALL,XRLENV_STATE_DB_PATH=${STATE_DB}" "${CONTROL_SBATCH}"
sleep 3

log "opening SSH tunnel to ${CP_NODE} (admin http://localhost:9080, metrics http://localhost:9190)"
# Drop any stale tunnel on these forwards, then open a fresh backgrounded one.
pkill -f "ssh -fN.*${CP_NODE}" 2>/dev/null || true
if ssh -fN "${SSH_OPTS[@]}" "${TUNNEL_FWDS[@]}" "${CP_NODE}"; then
    echo "    tunnel up → http://localhost:9080 (admin), http://localhost:9190 (metrics)"
else
    echo "    WARN: tunnel failed — the CP may still be starting. Re-run:"
    echo "      ssh -fN ${TUNNEL_FWDS[*]} ${CP_NODE}"
fi

# ── 3) sysbox pool: idempotent install (skip nodes already advertising it) ────
if [ "${#SYSBOX_POOL[@]}" -gt 0 ]; then
    log "configuring sysbox pool (idempotent): ${SYSBOX_POOL[*]}"
    for node in "${SYSBOX_POOL[@]}"; do
        if ssh "${SSH_OPTS[@]}" "${node}" 'docker info 2>/dev/null | grep -qi sysbox-runc'; then
            # Already present, and the bootstrap did NOT restart docker for it.
            # The docker-ready NodeHello gate makes the bootstrap-started agent
            # advertise sysbox-runc on its own — do NOT restart the agent here:
            # that races the still-in-progress node-job bootstrap (pip install +
            # service start) and wedges the process (Tasks:1 / 336K stub).
            echo "    ${node}: sysbox-runc already registered — skip (hello gate re-advertises)"
        else
            echo "    ${node}: installing sysbox..."
            # The repo + vendored binaries live on shared /shared-fs, reachable from
            # the node, so the installer runs in place (no scp needed).
            if ssh "${SSH_OPTS[@]}" "${node}" \
                    "sudo bash ${REPO_ROOT}/xrlenv_plugins/sysbox/install_sysbox_node.sh"; then
                echo "    ${node}: OK"
                # The install just restarted docker AFTER the bootstrap started the
                # agent, so its cached runtime set is stale and `Requires=` (not
                # `BindsTo=`) docker.service won't re-probe. Restart the agent to
                # re-advertise — but only once the bootstrap has finished STARTING
                # it (wait for the unit to be active), else we race that start and
                # wedge the process the same way.
                echo "    ${node}: waiting for the agent to settle, then re-advertising..."
                # Only restart once the bootstrap has the unit ACTIVE — if it
                # never comes active (a slow or wedged bootstrap) we must NOT
                # restart into that, so skip + warn rather than fall through.
                # Distinct exit codes so the local warning tells the two failure
                # modes apart: 2 = never became active; other = the restart itself
                # failed after it was active (SSH/sudo/systemd).
                if ssh "${SSH_OPTS[@]}" "${node}" '
                        active=0
                        for _ in $(seq 1 30); do
                            if systemctl is-active --quiet xrlenv-node; then
                                active=1; break
                            fi
                            sleep 5
                        done
                        if [ "$active" != 1 ]; then
                            echo "agent never became active within the wait window" >&2
                            exit 2
                        fi
                        sudo systemctl restart xrlenv-node'; then
                    echo "    ${node}: node agent restarted (re-advertised)"
                else
                    rc=$?
                    if [ "$rc" = 2 ]; then
                        echo "    ${node}: WARN — agent never became active; re-advertise skipped"
                    else
                        echo "    ${node}: WARN — agent restart failed after it became active (rc=${rc})"
                    fi
                fi
            else
                echo "    ${node}: install FAILED (see output above)"
            fi
        fi
    done
else
    log "no sysbox pool configured (SYSBOX_POOL empty) — skipping sysbox setup"
fi

# ── 4) P6 CPU-isolation pool: idempotent enable (skip nodes already cgroupfs) ──
if [ "${#CPU_ISOLATION_POOL[@]}" -gt 0 ]; then
    log "configuring CPU-isolation pool (idempotent): ${CPU_ISOLATION_POOL[*]}"
    for node in "${CPU_ISOLATION_POOL[@]}"; do
        # Runc-only overlap guard: flipping a sysbox node to cgroupfs can break
        # its DinD workloads (P6 v1 only proves runc). Warn, but proceed.
        for _sb in "${SYSBOX_POOL[@]:-}"; do
            [ "${node}" = "${_sb}" ] && echo "    ${node}: WARN — also in SYSBOX_POOL; cgroupfs may break sysbox DinD (CPU isolation is runc-only)"
        done
        if ssh "${SSH_OPTS[@]}" "${node}" 'docker info --format "{{.CgroupDriver}}" 2>/dev/null | grep -qx cgroupfs'; then
            echo "    ${node}: docker already on cgroupfs — running enable (idempotent, no docker bounce)"
        else
            echo "    ${node}: enabling CPU isolation (flips docker to cgroupfs → restarts docker + agent)..."
        fi
        # enable_cpu_isolation.sh lives on shared /shared-fs (reachable from the node);
        # it restarts docker + the agent when it flips the driver. Same race as
        # sysbox: don't restart into a still-bootstrapping agent — wait for the
        # unit to be ACTIVE first, then run the enable. Distinct exit codes: 2 =
        # agent never became active; other = the enable script itself failed.
        if ssh "${SSH_OPTS[@]}" "${node}" '
                active=0
                for _ in $(seq 1 30); do
                    if systemctl is-active --quiet xrlenv-node; then active=1; break; fi
                    sleep 5
                done
                if [ "$active" != 1 ]; then
                    echo "agent never became active within the wait window" >&2
                    exit 2
                fi
                sudo bash '"${REPO_ROOT}"'/scripts/enable_cpu_isolation.sh'; then
            echo "    ${node}: CPU isolation enabled — agent re-runs its self-test at NodeHello"
        else
            rc=$?
            if [ "$rc" = 2 ]; then
                echo "    ${node}: WARN — agent never became active; CPU-isolation enable skipped"
            else
                echo "    ${node}: WARN — enable_cpu_isolation.sh failed (rc=${rc}); node stays non-capable"
            fi
        fi
    done
    echo "    (check 'xrlenv nodes' CPU_ISOLATION column — expect 'yes …' once reconnected)"
else
    log "no CPU-isolation pool configured (CPU_ISOLATION_POOL empty) — skipping"
fi

# ── 5) post-bootstrap verification: catch nodes whose bootstrap silently died ──
# A node whose srun bootstrap task exits non-zero (e.g. a flaky docker restart
# during data-root relocation) is NOT surfaced by the RUNNING check above — it
# keeps the PRIOR deploy's /etc/xrlenv/node.env, dials a dead control plane, and
# shows up "absent" with no other signal. Sweep every allocated node and assert
# its agent config points at THIS control plane; WARN loudly on any mismatch.
# Two proofs, each FATAL unless XRLENV_DEPLOY_FORCE=1:
#  (a) config — every allocated node's /etc/xrlenv/node.env points at THIS CP (a
#      failed bootstrap keeps a stale env, dials a dead CP, shows up "absent");
#  (b) LIVE   — every allocated node is 'connected' in the CP registry (a current
#      NodeHello), not just correctly configured.
log "verifying node agent config + live registration across the allocation..."
# state.db is CP-box-local → the registration check must read it ON the CP box (ssh), since
# it isn't on the shared FS anymore. deploy_verify_fleet honors this env; unset = read directly.
export XRLENV_VERIFY_CP_SSH_HOST="${CP_NODE}"
deploy_verify_fleet "${NODE_JOB}" "${CP_NODE}:${CP_PORT}" "${STATE_DB}" \
    "${REPO_ROOT}/.venv/bin/xrlenv"   # fail-closed (XRLENV_DEPLOY_FORCE=1 overrides)

log "deploy complete."
