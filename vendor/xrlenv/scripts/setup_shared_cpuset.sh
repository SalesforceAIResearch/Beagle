#!/usr/bin/env bash
# setup_shared_cpuset.sh — (re)establish the delegated P6 shared-parent cpuset
# cgroup ``/sys/fs/cgroup/xrlenv-shared`` for the NON-ROOT node agent (§8.13).
#
# Idempotent + no-op-safe. Called by:
#   * scripts/enable_cpu_isolation.sh, once the container probe has proven the
#     node honors cgroup_parent cpuset propagation, and
#   * the ``xrlenv-cpu-isolation.service`` systemd oneshot AT BOOT — because
#     cgroups do NOT survive a reboot, so ``xrlenv-shared`` + its chown-delegation
#     must be recreated after docker is up and BEFORE the agent starts, or the
#     node reverts to non-capable until a redeploy re-runs the enable script.
#
# Does NOTHING unless docker is on the ``cgroupfs`` cgroup driver (the P6 gate),
# so it's safe to leave the boot unit enabled on a node later reverted to
# systemd-driver. Never flips the driver, runs the probe, or restarts anything —
# it only creates + delegates the cgroup (the one privileged step the non-root
# agent cannot do for itself).
#
# Env knobs: CGROUP_ROOT (default /sys/fs/cgroup), AGENT_USER (default: the
# xrlenv-node unit's User=, else ``xrlenv``).
set -euo pipefail

CGROUP_ROOT="${CGROUP_ROOT:-/sys/fs/cgroup}"
SHARED="${CGROUP_ROOT}/xrlenv-shared"

# P6 gate: only a cgroupfs-driver node is capable. On anything else (a stock
# systemd-driver node, or one reverted after enable) this is a clean no-op.
DRIVER="$(docker info --format '{{.CgroupDriver}}' 2>/dev/null || echo unknown)"
if [ "${DRIVER}" != "cgroupfs" ]; then
    echo "setup_shared_cpuset: docker cgroup driver is '${DRIVER}', not cgroupfs"
    echo "                     — CPU isolation not enabled on this node; nothing to do."
    exit 0
fi

# The agent user to delegate to. Read from the unit file's ``User=`` (works even
# before the agent is started — the boot-time case), defaulting to ``xrlenv``.
AGENT_USER="${AGENT_USER:-$(systemctl show xrlenv-node -p User --value 2>/dev/null || true)}"
AGENT_USER="${AGENT_USER:-xrlenv}"

echo "setup_shared_cpuset: (re)establishing ${SHARED}, delegating to '${AGENT_USER}'"

# Root cgroup must expose cpuset to its children so xrlenv-shared has cpuset.cpus.
if ! grep -qw cpuset "${CGROUP_ROOT}/cgroup.subtree_control" 2>/dev/null; then
    echo "+cpuset" > "${CGROUP_ROOT}/cgroup.subtree_control"
fi
mkdir -p "${SHARED}"
# Enable cpuset for xrlenv-shared's CHILDREN (the containers) — idempotent.
echo "+cpuset" > "${SHARED}/cgroup.subtree_control" 2>/dev/null || true
# Seed the no-pinning state: the shared pool is every online CPU (the agent's
# ledger overwrites this with the live complement once it wires the parent).
ALL="$(cat "${CGROUP_ROOT}/cpuset.cpus.effective" 2>/dev/null || true)"
[ -n "${ALL}" ] && echo "${ALL}" > "${SHARED}/cpuset.cpus"
# chown the delegation surface: the dir + the exact files the agent writes.
chown "${AGENT_USER}" "${SHARED}" 2>/dev/null || true
for f in cpuset.cpus cgroup.procs cgroup.subtree_control cgroup.threads; do
    [ -e "${SHARED}/${f}" ] && chown "${AGENT_USER}" "${SHARED}/${f}" 2>/dev/null || true
done
echo "setup_shared_cpuset: ${SHARED} ready (cpuset.cpus=$(cat "${SHARED}/cpuset.cpus" 2>/dev/null || echo '?'), owner=$(stat -c '%U' "${SHARED}/cpuset.cpus" 2>/dev/null || echo '?'))"
