#!/usr/bin/env bash
# Skeleton: per-task image build script for Pattern-A benchmarks.
#
# Pattern-A plug-ins (terminal-bench-2 is the reference) build a
# per-task image overlay on each VM at provisioning time. The xrlenv
# operator runs this script during ``deploy/bring-up-node.sh`` to
# pre-warm every task's image so first-rollout latency is not
# dominated by ``docker build``.
#
# This skeleton is a no-op — keep it iff your plug-in is Pattern-A,
# delete the file otherwise. The skeleton's pyproject.toml does NOT
# force-include this directory into the wheel (operators run it from
# a checkout), so removing the file leaves no stale packaging
# reference behind.

set -euo pipefail

echo "example_bench: no per-task images to build (skeleton)"
