#!/usr/bin/env bash
# bootstrap-aws.sh — thin wrapper around ``xrlenv bootstrap --target aws``.
#
# The bash bootstrap logic was replaced with a Python subcommand at
# ``xrlenv/cli/bootstrap.py``. This
# wrapper preserves the historical operator interface:
#
#     sudo -E bash deploy/bootstrap-aws.sh [--hyperpod] [<control-plane>] [<node-id>]
#
# All knobs documented under ``xrlenv bootstrap --help`` are also
# settable via env var (XRLENV_CONTROL_PLANE / XRLENV_NODE_ID /
# XRLENV_WHEEL / XRLENV_REPO / XRLENV_VERSION / XRLENV_NODE_TOKEN).
# Pass --target-os to override the /etc/os-release probe on custom
# AMIs.
#
# --hyperpod (SageMaker HyperPod only): before installing/starting
# Docker, relocate Docker's data-root onto the node's large EBS volume
# (mounted at /opt/sagemaker) so image layers land on ~500 GB of block
# storage instead of the ~97 GB root disk. Without this, HyperPod nodes
# default to /var/lib/docker on the root disk and fill it. Consumed by
# this wrapper; NOT forwarded to the Python bootstrap.

set -euo pipefail

# Extract the HyperPod-only --hyperpod flag from anywhere in the args.
HYPERPOD=0
_args=()
for _a in "$@"; do
    if [ "$_a" = "--hyperpod" ]; then HYPERPOD=1; else _args+=("$_a"); fi
done
set -- ${_args[@]+"${_args[@]}"}

if (( $# >= 1 )); then export XRLENV_CONTROL_PLANE="$1"; shift; fi
if (( $# >= 1 )); then export XRLENV_NODE_ID="$1"; shift; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# HyperPod: relocate Docker's data-root onto the EBS volume BEFORE the
# Python bootstrap installs/starts Docker, so the very first pulls land
# on /opt/sagemaker. The helper is idempotent and guarded — it aborts if
# the target is Lustre or resolves to the root device.
if (( HYPERPOD )); then
    echo "==> --hyperpod: relocating Docker data-root to the EBS volume (/opt/sagemaker)"
    bash "${REPO_ROOT}/deploy/node/set_docker_data_root.sh"
fi

# Optional CLIENT config: point THIS WORKER's Docker daemon at the xrlenv
# pull-through registry mirror so cold / re-pulls go over the internal network
# instead of Docker Hub. This only edits daemon.json (registry-mirrors) — it does
# NOT run a registry here; the mirror SERVER runs separately on the control-plane
# box or a dedicated registry node (deploy/registry/run-registry-mirror.sh), never
# on a worker. On a fresh node Docker isn't installed yet, so this just writes
# daemon.json and dockerd reads it on first start; on a re-bootstrap where Docker
# is already running, configure_docker_registry.sh live-reloads it so the mirror
# takes effect without a stale-config gap. Set
# XRLENV_REGISTRY_MIRROR=http://<mirror-ip>:5010 to enable; unset leaves the node
# pulling straight from Docker Hub.
#
# XRLENV_PRIVATE_REGISTRY=<registry-ip>:5011 (optional) does the analogous one-time
# wiring for the PRIVATE (writable) registry — it adds that host to
# insecure-registries so this node can pull (or, if it's also a build host, push)
# our own built images over plain HTTP. Unlike the mirror it is NOT a
# registry-mirrors entry: private images are addressed by named ref. Setting it
# here means a freshly provisioned node can pull the private set with no manual
# per-node step. Full setup: deploy/registry/README.md or the Sphinx registry pages.
#
# XRLENV_SCRATCH_REGISTRY=<registry-ip>:5012 (optional) does the same for the
# SCRATCH (build-on-demand) registry — every node running scratch_build rollouts
# push/pulls the on-demand-built image there over plain HTTP, so it must be in
# insecure-registries too.
if [ -n "${XRLENV_REGISTRY_MIRROR:-}" ] || [ -n "${XRLENV_PRIVATE_REGISTRY:-}" ] || [ -n "${XRLENV_SCRATCH_REGISTRY:-}" ]; then
    [ -n "${XRLENV_REGISTRY_MIRROR:-}" ] && echo "==> registry mirror (client): ${XRLENV_REGISTRY_MIRROR}"
    [ -n "${XRLENV_PRIVATE_REGISTRY:-}" ] && echo "==> private registry (client): ${XRLENV_PRIVATE_REGISTRY}"
    [ -n "${XRLENV_SCRATCH_REGISTRY:-}" ] && echo "==> scratch registry (client): ${XRLENV_SCRATCH_REGISTRY}"
    MIRROR_URL="${XRLENV_REGISTRY_MIRROR:-}" \
    PRIVATE_REGISTRY="${XRLENV_PRIVATE_REGISTRY:-}" \
    SCRATCH_REGISTRY="${XRLENV_SCRATCH_REGISTRY:-}" \
        bash "${REPO_ROOT}/deploy/registry/configure_docker_registry.sh"
fi

# ── CPU isolation (opt-in) ───────────────────────────────────────────────────
# Hard CPU isolation lets the
# node confine unpinned containers to the complement of the pinned cores via a
# shared-parent cpuset cgroup. A node advertises this capability
# (``isolation_capable=true``) only after a real self-test proves its docker +
# cgroup driver honor ``cgroup_parent`` cpuset propagation — and that self-test
# is gated (v1) on the ``cgroupfs`` docker cgroup driver. AL2023 / Ubuntu
# 22.04 default to the ``systemd`` driver, so a STOCK node stays NON-capable and
# behaves exactly as today (per-container pinning / CFS quota) — no action here
# changes that.
#
# To make a node isolation-capable, run (as root, on the worker, in a
# maintenance window — it restarts docker + the agent, bouncing containers):
#     sudo bash deploy/../deploy/node/enable_cpu_isolation.sh
# which flips docker to the cgroupfs driver, builds a long-lived probe image, and
# persists XRLENV_SELFTEST_IMAGE in /etc/xrlenv/cpu_isolation.env (survives
# bootstrap rewrites; delete it + restart docker to revert). Validated end-to-end
# on a real node 2026-07-28 (an unpinned container becomes physically unable to
# reach a pinned core). The self-test validates the DEFAULT runtime (runc) only;
# it does not prove sysbox.
#
# Set XRLENV_ENABLE_CPU_ISOLATION=1 in the bootstrap env to run that script
# automatically. It runs AFTER the Python bootstrap below (not before) — the
# Python bootstrap is what installs + starts Docker on a fresh node, and the
# enable script needs a running Docker to build the probe image and restart the
# daemon. Running it first would fail on a fresh node and leave it non-capable.

# See deploy/bootstrap-gcp.sh for why we invoke the bootstrap module
# as a flat script (avoids importing xrlenv.__init__ on fresh VMs
# where pydantic isn't installed yet). NOT ``exec`` (unlike gcp) so the shell
# survives to run the optional CPU-isolation auto-enable hook below after Docker is
# installed; ``set -e`` still aborts here if the bootstrap itself fails.
python3 "${REPO_ROOT}/xrlenv/cli/bootstrap.py" \
    --target aws \
    --xrlenv-repo "${REPO_ROOT}" \
    "$@"

# CPU isolation (opt-in) — now Docker is installed + the agent set up. The
# driver flip + agent restart re-establishes the node's connection as capable.
if [ "${XRLENV_ENABLE_CPU_ISOLATION:-0}" = "1" ]; then
    echo "==> XRLENV_ENABLE_CPU_ISOLATION=1: enabling CPU isolation on this node"
    sudo -E bash "${REPO_ROOT}/deploy/node/enable_cpu_isolation.sh" || {
        echo "WARN: enable_cpu_isolation.sh failed; node stays non-capable" >&2
    }
fi
