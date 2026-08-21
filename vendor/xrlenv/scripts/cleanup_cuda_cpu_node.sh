#!/usr/bin/env bash
# cleanup_cuda_cpu_node.sh — Remove CUDA toolkits from CPU-only HyperPod nodes.
#
# The Deep Learning AMI ships ~41 GB of CUDA toolkits under /usr/local/
# that are useless on CPU instances (m7i, c7i, etc.). Removing them
# frees local root disk for containerd staging during image pulls.
#
# Usage (on each CPU node, via sudo):
#   sudo bash /path/to/xrlenv/scripts/cleanup_cuda_cpu_node.sh

set -euo pipefail

echo "==> Pre-cleanup disk usage:"
df -h /

FREED=0

for dir in /usr/local/cuda-*/; do
    [ -d "$dir" ] || continue
    SIZE=$(du -sm "$dir" 2>/dev/null | awk '{print $1}')
    echo "==> Removing $dir (${SIZE:-?} MB)"
    rm -rf "$dir"
    FREED=$((FREED + ${SIZE:-0}))
done

# Remove the /usr/local/cuda symlink (points to default version)
if [ -L /usr/local/cuda ]; then
    echo "==> Removing symlink /usr/local/cuda"
    rm -f /usr/local/cuda
fi

echo "==> Freed ~${FREED} MB"
echo "==> Post-cleanup disk usage:"
df -h /
