#!/usr/bin/env bash
#
# Why this script lives per-package (see also
# byo_dataset_harbor/scripts/build-task-images.sh): the two build
# scripts share the xrlenv-CLI lookup + the stub-runtime layer
# invocation pattern, but diverge on cache walking, tag templates,
# and Dockerfile shape. A shared helper would obscure those
# differences. Acceptable duplication.
#
# echo_bench — build per-instance Docker images.
#
# All 3 echo_bench instances share the same Dockerfile; the
# build script tags the result three times so the resolver's
# per-instance image refs (``echo-bench/<instance_id>:0.1``)
# resolve at sandbox-create time.
#
# Total image footprint: ~150 MB shared across the three tagged
# layers (python:3.12-slim + xrlenv stub-runtime additions). Build
# time on a warm Docker daemon: ~10 s for the base + ~1 s per
# additional tag.
#
# Usage:
#   bash scripts/build-task-images.sh [--force]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE_DIR="${PACKAGE_ROOT}/xrlenv_plugins/benchmarks/echo_bench/scripts"
INSTANCE_IDS=(echo-hello echo-multiline echo-symbols)
BASE_TAG="echo-bench-base:0.1"

# Pick the xrlenv CLI: env override → activated venv → walk up looking
# for .venv/bin/xrlenv. The build script needs ``xrlenv stub-runtime
# layer``; require it to be reachable.
if [ -n "${XRLENV_BIN:-}" ]; then
    XRLENV="${XRLENV_BIN}"
elif command -v xrlenv >/dev/null 2>&1; then
    XRLENV="$(command -v xrlenv)"
else
    candidate="${PACKAGE_ROOT}/../../../.venv/bin/xrlenv"
    if [ -x "${candidate}" ]; then
        XRLENV="${candidate}"
    else
        echo "build-task-images: 'xrlenv' CLI not found. Activate the" >&2
        echo "venv (source .venv/bin/activate) or pass" >&2
        echo "XRLENV_BIN=/path/to/xrlenv." >&2
        exit 2
    fi
fi

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

# Stage 1 — build the base image once (deduplicated; same Dockerfile
# for all instances). Using docker's normal layer cache; --force
# triggers a no-cache rebuild.
echo ">>> building base image ${BASE_TAG}"
# ``--label org.xrlenv.*`` marks the base layer as an xrlenv-built
# build intermediate so the admin /images view filters it out by
# default. Final tags inherit ``role=final`` from
# ``Dockerfile.stub-runtime`` via ``xrlenv stub-runtime layer``.
if [ "$FORCE" = "1" ]; then
    docker build --no-cache \
        --label "org.xrlenv.owned=true" \
        --label "org.xrlenv.role=intermediate" \
        -t "${BASE_TAG}" \
        -f "${DOCKERFILE_DIR}/Dockerfile" \
        "${DOCKERFILE_DIR}"
else
    docker build \
        --label "org.xrlenv.owned=true" \
        --label "org.xrlenv.role=intermediate" \
        -t "${BASE_TAG}" \
        -f "${DOCKERFILE_DIR}/Dockerfile" \
        "${DOCKERFILE_DIR}"
fi

# Stage 2 — apply the xrlenv stub-runtime layer + tag per-instance.
# The stub-runtime layer adds whatever the platform needs to run
# ``python3 -m xrlenv.sandbox_stub`` (today: just the existence of
# python3, which the python:3.12-slim base already supplies — so
# the layer is essentially a re-tag).
for inst in "${INSTANCE_IDS[@]}"; do
    final_tag="echo-bench/${inst}:0.1"
    if [ "$FORCE" = "0" ] && docker image inspect "${final_tag}" >/dev/null 2>&1; then
        echo ">>> ${final_tag} already built; skip (use --force to rebuild)"
        continue
    fi
    echo ">>> stub-runtime layer: ${BASE_TAG} -> ${final_tag}"
    "${XRLENV}" stub-runtime layer --base "${BASE_TAG}" --out "${final_tag}"
done

echo
echo "Built ${#INSTANCE_IDS[@]} image(s):"
for inst in "${INSTANCE_IDS[@]}"; do
    echo "  echo-bench/${inst}:0.1"
done
echo
echo "Run the smoke:"
echo "  .venv/bin/python examples/pip_new_datasets_or_benchmark/echo_bench/examples/echo_smoke.py --local"
