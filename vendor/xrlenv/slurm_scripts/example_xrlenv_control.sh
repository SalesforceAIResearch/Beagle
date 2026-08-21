#!/bin/bash
#
#SBATCH --job-name=example-xrlenv-control                # Job name
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --nodelist=node-host
#SBATCH --output=/path/to/xrlenv/slurm_logs/%x_%j.out         # Standard output log
#SBATCH --error=/path/to/xrlenv/slurm_logs/%x_%j.err          # Standard error log
#SBATCH --partition=your-slurm-partition
#SBATCH --account=your-slurm-account

echo "Control plane IP: $(hostname -I)"

cd /path/to/xrlenv
source .venv/bin/activate
which python


# Regenerate the roster from the worker Slurm script so the SBATCH
# nodelist stays the single source of truth. Override either path when
# running a different worker script or writing a different inventory.
#
# Sysbox pool (optional): XRLENV_SYSBOX_POOL is a space-separated list of
# hostnames/ids (set by the deploy orchestrator, deploy_example.sh) to mark
# `sysbox: true` + permit `sysbox-runc` in policy.allowed_runtimes at
# generation time. Empty (the default) is a no-op — the ordinary roster.
# Actually INSTALLING Sysbox on those nodes is a separate idempotent step in
# the deploy script; there is no hard ordering dependency because the live
# NodeHello runtime advertisement — not this marker — gates placement. Markers
# already in the destination file are preserved regardless. See
# xrlenv_plugins/sysbox/README.md "Install order".
XRLENV_NODE_SLURM_SCRIPT=./slurm_scripts/example_xrlenv_node.sh
XRLENV_NODES_YAML=./slurm_scripts/example_hyperpod_nodes.yaml
_sysbox_args=()
for _n in ${XRLENV_SYSBOX_POOL:-}; do _sysbox_args+=(--sysbox-node "$_n"); done
[ "${#_sysbox_args[@]}" -gt 0 ] && _sysbox_args+=(--allowed-runtime sysbox-runc)
# Per-node sysbox concurrency cap (sysbox-fs wedge prevention). Preserved per
# node across regen; this stamps the default onto pool nodes lacking one.
[ -n "${XRLENV_SYSBOX_MAX_CONCURRENT:-}" ] && \
    _sysbox_args+=(--sysbox-max-concurrent "${XRLENV_SYSBOX_MAX_CONCURRENT}")
# Allowed host bind paths (EvoClaw golden/data-root, read-only) — one
# --allowed-host-path per non-empty line, additively merged into policy.
while IFS= read -r _hp; do
    [ -n "$_hp" ] && _sysbox_args+=(--allowed-host-path "$_hp")
done <<< "${XRLENV_ALLOWED_HOST_PATHS:-}"
xrlenv nodes-from-slurm \
    --slurm-script "${XRLENV_NODE_SLURM_SCRIPT}" \
    --output "${XRLENV_NODES_YAML}" \
    ${_sysbox_args[@]+"${_sysbox_args[@]}"}

lscpu | grep -E 'CPU\(s\)|Core\(s\) per socket|Socket\(s\)|Thread\(s\) per core'

free -h


# Firehose log goes to a STABLE, size-rotating file (50 MiB × 10 = ~500 MiB
# ceiling) rather than the per-job Slurm --output capture, which has no
# rotation and grows without bound for a long-running control plane. The path
# is stable across restarts, so `tail -f` always targets the same file instead
# of hunting for the current job id. With --log-file set, stdout (the Slurm
# %x_%j.out below) keeps only WARNING+, so it stays small and still shows crashes.
XRLENV_CONTROL_LOG=~/.xrlenv/xrlenv-up-control.log

# ---- Co-located registry: probe it over loopback ----------------------------
# The tag->digest resolver (freshness model) probes the registry FROM this
# control-plane process. When the registry runs ON this same box, the box can
# reach its own docker-published port over loopback (127.0.0.1) but typically
# NOT via its own external name/IP — host->own-port hairpin NAT is unreliable
# (notably under docker 29). Remote worker nodes reach the external name fine,
# so the image ref must keep it; we redirect only the control plane's manifest
# probe to loopback. The returned digest ref is unchanged, so nodes still pull
# the external address + pinning is preserved. Built from THIS box's own names
# so it's self-correcting: this cluster points at the SHARED registry on another box, so
# no image ref names this box as its registry host and these entries never
# match (harmless) — the CP dials the shared registry's external name directly,
# which works cross-box. Covers private (:5011) + mirror (:5010).
_hostmap=""
for _n in "$(hostname -s)" "$(hostname -I | awk '{print $1}')"; do
    for _p in 5011 5010; do
        _hostmap="${_hostmap:+$_hostmap,}${_n}:${_p}=127.0.0.1:${_p}"
    done
done
export XRLENV_REGISTRY_RESOLVE_HOST_MAP="$_hostmap"
echo "registry resolve host-map: $XRLENV_REGISTRY_RESOLVE_HOST_MAP"

# Enable Python's fault handler so a FATAL signal (e.g. SIGBUS from a SQLite
# mmap fault on a network-backed state.db) dumps the crashing thread's stack to
# stderr (the Slurm .err) instead of dying silently. Cheap insurance; it only
# fires on fatal signals (SIGSEGV/SIGBUS/SIGFPE/SIGABRT).
export PYTHONFAULTHANDLER=1

# state.db on CP-box-LOCAL disk (NOT Lustre). Previously it lived on /shared-fs and had to run
# TRUNCATE (rollback-journal): WAL's mmap'd -shm faults with a FATAL SIGBUS on a Lustre
# hiccup (took down both clusters 2026-07-30). But TRUNCATE on Lustre is ~38 ms/commit,
# and the CP commits synchronously in its event loop — under high concurrency those blocking
# commits serialize the loop and node commands (list_sandbox_ids/exec/heartbeat) time out.
# Moving state.db to local disk fixes BOTH: no network-backed mmap → WAL is SIGBUS-safe, and
# WAL on local disk is ~0.01 ms/commit (non-blocking reads too). Only state.db moves; the rest
# of $XRLENV_HOME (secrets/, runs/, config) stays on /shared-fs. It lives on the box's local 500 GB
# NVMe (/opt/sagemaker — the same volume docker's data-root uses), which survives reboots and CP
# restarts; it's lost only if the box is stopped/reprovisioned (acceptable: that loses the fleet
# too, and state rebuilds on restart). NOT /tmp — /tmp is systemd-tmpfiles-cleared. deploy_example.sh
# pre-creates this dir user-owned (sudo mkdir + chown), so the mkdir below is an idempotent no-op.
# Redis StateStore (phase 1) remains the long-term high-throughput answer.
# deploy_example.sh is the single source: it propagates its STATE_DB into this job via
# `sbatch --export=ALL,XRLENV_STATE_DB_PATH=...`, so this honors that value. The default
# below applies ONLY to a bare `sbatch example_xrlenv_control.sh` (no deploy) — keep it in sync
# with deploy_example.sh's STATE_DB (a mismatch there only affects a manual, non-deploy restart).
export XRLENV_STATE_DB_PATH="${XRLENV_STATE_DB_PATH:-/opt/sagemaker/xrlenv/state.db}"
mkdir -p "$(dirname "$XRLENV_STATE_DB_PATH")" 2>/dev/null || true
export XRLENV_SQLITE_JOURNAL_MODE=WAL

# `exec` so this Python process REPLACES the bash script and becomes the job
# step's main process. scancel's SIGTERM then reaches `xrlenv up` directly,
# which runs its graceful shutdown — drain in-flight rollouts, then close the
# state store, which checkpoints the SQLite WAL and removes state.db-{wal,shm}.
# Without exec, bash is the job's main process and does NOT forward SIGTERM to
# this child, so the control plane is SIGKILL'd and leaves an un-checkpointed
# state.db-wal behind. (No `sleep` keepalive needed: `xrlenv up` blocks until
# signalled.)
exec xrlenv up \
    --log-file "${XRLENV_CONTROL_LOG}" \
    --grpc-host node-host  --grpc-port 50051 \
    --admin-host 0.0.0.0 --admin-port 8080 --admin-allow-public \
    --admin-nodes-yaml "${XRLENV_NODES_YAML}" \
    --adaptive-admission --aimd-initial-limit 64 --aimd-p95-threshold-s 900 --aimd-max-limit 80 \
    --audit-retention-days 7 --raw-rollout-retention-days 180
# Retention (overrides the spec-20 defaults of audit=30 / raw_rollouts=14):
#   audit=7   — the audit table is what bloats state.db (per-auth rows); with
#               auth.token_used auditing off it barely grows, and a tight 7-day
#               window keeps the DB small on the Lustre-backed store (the SIGBUS
#               root cause was a large WAL on Lustre — keep state.db lean).
#   raw_rollouts=180 — rollout metadata is TINY (~20k rows ≈ a few MB) and is the
#               per-user accounting source behind /users, so keep ~6 months of
#               history instead of silently dropping it after 14 days.

# on the login node — exposes control-plane:8080 at login-node:9080 (shared)
# scancel --name=example-xrlenv-control && sbatch slurm_scripts/example_xrlenv_control.sh && ssh -fN -L 9080:localhost:8080 -L 9190:localhost:9090 node-host