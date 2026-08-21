#!/usr/bin/env bash
# set_docker_data_root.sh — point Docker's data-root at the node's large
# EBS volume (HyperPod mounts it at /opt/sagemaker) so image layers land
# on ~500 GB of fast block storage instead of the ~97 GB root disk.
#
# Why not FSx: overlay/overlayfs CANNOT mount on Lustre, so a data-root
# on /shared-fs silently falls back to the root disk — the "data-root was a
# lie" failure. /opt/sagemaker is a real XFS block device (EBS), so
# overlay mounts and Docker's bundled containerd snapshotter writes the
# real layers under data-root.
#
# We set ONLY data-root — matching the known-good node. We do NOT force
# storage-driver (forcing overlay2 on FSx is what broke things before);
# Docker's default image store writes under data-root once it is on a
# real block device.
#
# Safe to call two ways:
#   * on a running node (full stop / reconfigure / restart), or
#   * early in bootstrap before Docker is installed (just places the
#     config; Docker reads it on first start).
#
# Usage (on each node, via sudo):
#   sudo bash scripts/set_docker_data_root.sh
# Override the target if your EBS volume is mounted elsewhere:
#   sudo DATA_ROOT=/mnt/ebs/docker/data-root bash scripts/set_docker_data_root.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/opt/sagemaker/docker/data-root}"
DAEMON_JSON="/etc/docker/daemon.json"
MOUNT="$(dirname "$(dirname "$DATA_ROOT")")"   # e.g. /opt/sagemaker

# Merge existing daemon.json keys, set ONLY data-root. While doing that,
# scrub stale empty registry entries from earlier bad runs; Docker refuses
# to start when ``registry-mirrors`` contains values like ``http://:``.
write_daemon_json() {
    mkdir -p "$DATA_ROOT"
    python3 - "$DAEMON_JSON" "$DATA_ROOT" <<'PY'
import json, os, sys
from urllib.parse import urlparse

path, data_root = sys.argv[1], sys.argv[2]
cfg = {}
if os.path.isfile(path):
    with open(path) as f:
        cfg = json.load(f)

def _clean_registry_mirrors(values):
    if not isinstance(values, list):
        return values
    clean = []
    for value in values:
        if not isinstance(value, str):
            continue
        parsed = urlparse(value.strip())
        if parsed.scheme in ("http", "https") and parsed.netloc:
            clean.append(value.strip())
    return clean

def _clean_insecure_registries(values):
    if not isinstance(values, list):
        return values
    return [
        value.strip()
        for value in values
        if isinstance(value, str) and value.strip() not in ("", ":")
    ]

if "registry-mirrors" in cfg:
    cfg["registry-mirrors"] = _clean_registry_mirrors(cfg["registry-mirrors"])
if "insecure-registries" in cfg:
    cfg["insecure-registries"] = _clean_insecure_registries(cfg["insecure-registries"])
cfg["data-root"] = data_root
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
    echo "==> ${DAEMON_JSON}:"; cat "$DAEMON_JSON"
}

# 1. Guard: the target must live on a real, non-root, overlay-capable
#    block filesystem. This is what prevents a repeat of the FSx debacle.
src_fstype="$(findmnt -no FSTYPE --target "$MOUNT" 2>/dev/null || echo '')"
root_dev="$(findmnt -no SOURCE / 2>/dev/null || echo '')"
tgt_dev="$(findmnt -no SOURCE --target "$MOUNT" 2>/dev/null || echo '')"
case "$src_fstype" in
    ext4|xfs) : ;;  # overlay-capable
    lustre|nfs|fuse.*|'')
        echo "ERROR: '$MOUNT' is fstype '${src_fstype:-unknown}' — overlay cannot mount there." >&2
        echo "       Point DATA_ROOT at a real block device (EBS). Aborting." >&2
        exit 1 ;;
    *)
        echo "WARN: '$MOUNT' fstype '$src_fstype' is unusual — verify overlay support before relying on this." >&2 ;;
esac
if [ -n "$tgt_dev" ] && [ "$tgt_dev" = "$root_dev" ]; then
    echo "ERROR: '$MOUNT' resolves to the root device ($root_dev) — that's the small disk we're avoiding. Aborting." >&2
    exit 1
fi
echo "==> Target data-root: $DATA_ROOT  (mount=$MOUNT fstype=$src_fstype dev=$tgt_dev)"

# 2. If Docker isn't installed yet (called early in bootstrap), just place
#    the config — Docker will read it on first start — and stop here.
if ! command -v docker >/dev/null 2>&1; then
    write_daemon_json
    echo "==> docker not installed yet; data-root config placed (applies on first start)."
    exit 0
fi

# 2b. Idempotency guard: if Docker is ALREADY running on the target data-root,
#     this is a re-bootstrap of an already-relocated node. Refresh the config
#     file (scrubs stale/empty registry entries) but SKIP the stop/start — that
#     restart is the disruptive, occasionally-flaky step: a spurious "Job for
#     docker.service canceled" on `systemctl start docker` here aborts the whole
#     bootstrap under `set -e`, which strands the node on the PRIOR deploy's
#     /etc/xrlenv/node.env (dials a dead control plane → shows up "absent").
#     Nothing to relocate means nothing to restart.
CURRENT_ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo UNKNOWN)"
if [ "$CURRENT_ROOT" = "$DATA_ROOT" ]; then
    echo "==> Docker already on data-root $DATA_ROOT — refreshing config, skipping restart."
    write_daemon_json
    echo "==> Disk:"; df -h / "$MOUNT"
    exit 0
fi

# 3. Running node NOT yet on the target: stop Docker (+ node agent) before
#    relocating storage.
echo "==> Stopping Docker..."
systemctl stop docker.socket docker.service 2>/dev/null || true
if systemctl is-active --quiet xrlenv-node 2>/dev/null; then
    echo "==> Stopping xrlenv-node..."
    systemctl stop xrlenv-node
fi

# 4. Write the config and restart Docker. Retry the start: on a busy node the
#    first `systemctl start docker` can lose to a competing systemd job and come
#    back "Job for docker.service canceled" — a transient, not a config error.
#    `reset-failed` + a few backed-off attempts turns that flake into a settle
#    instead of a hard bootstrap abort (which used to strand the node "absent").
write_daemon_json
echo "==> Starting Docker..."
start_docker() {
    local i
    for i in 1 2 3 4 5; do
        systemctl reset-failed docker.service docker.socket 2>/dev/null || true
        if systemctl start docker.socket docker.service 2>/dev/null \
                && systemctl is-active --quiet docker.service; then
            return 0
        fi
        echo "==> docker start attempt $i did not settle; retrying in 3s..." >&2
        sleep 3
    done
    return 1
}
if ! start_docker; then
    echo "ERROR: docker.service failed to start after retries." >&2
    systemctl status docker.service --no-pager -l 2>&1 | tail -20 >&2 || true
    exit 1
fi

# 5. Verify the live root matches.
ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo UNKNOWN)"
DRIVER="$(docker info --format '{{.Driver}}' 2>/dev/null || echo UNKNOWN)"
echo "==> Docker data-root: $ROOT   (driver: $DRIVER)"
if [ "$ROOT" != "$DATA_ROOT" ]; then
    echo "ERROR: data-root is '$ROOT', expected '$DATA_ROOT'." >&2
    exit 1
fi
echo "==> Disk after switch:"; df -h / "$MOUNT"

# 6. Restart node agent if installed.
if systemctl is-enabled --quiet xrlenv-node 2>/dev/null; then
    echo "==> Restarting xrlenv-node..."
    systemctl start xrlenv-node
fi

cat <<EOF
==> Done. New pulls now land on $MOUNT (the EBS volume).
    The old store at /var/lib/docker is now unused. Once you've confirmed
    pulls grow $MOUNT (and / stays flat), reclaim the root disk with:
        sudo rm -rf /var/lib/docker
EOF
