#!/bin/bash
#
#SBATCH --job-name=xrlenv-build-push
#SBATCH --nodes=2
#SBATCH --nodelist=node-host,node-host
#SBATCH --output=/path/to/xrlenv/slurm_logs/%x_%j.out
#SBATCH --error=/path/to/xrlenv/slurm_logs/%x_%j.err
#SBATCH --partition=your-slurm-partition
#SBATCH --account=your-slurm-account
#
# One-shot, CLUSTER-AGNOSTIC build + push. This is NOT an xrlenv service — it does
# not talk to a control plane, needs no node token, and is unrelated to a specific cluster.
# It just builds every image in a build plan from its Dockerfile and pushes it to
# the registry, fanned out across the allocation (one shard per node, shard
# index/count auto-read from $SLURM_PROCID / $SLURM_NTASKS). Finite: the job exits
# when the build completes. Re-running skips images already in the registry, so a
# partial/failed run just resumes.
#
# Pick your build nodes with --nodes/--nodelist above (any nodes with docker + the
# shared FSx home — they don't have to be a particular cluster; keep --nodes in
# sync with the nodelist count). Override the plan / registry / concurrency below
# or in .env, then:
#
#   sbatch slurm_scripts/build_and_push_images.sh
#   tail -f slurm_logs/xrlenv-build-push_*.out
#   scancel --name=xrlenv-build-push



# Exported so srun propagates them into each per-node task. Without `export`,
# these would be invisible inside the single-quoted srun body below (it is the
# task's shell that expands $PROJECT_ROOT), and `cd "$PROJECT_ROOT"` would fail
# under `set -u`.
export PROJECT_ROOT=/path/to/xrlenv
export XRLENV_BUILD_PLAN=xrlenv_plugins/benchmarks/seta/build_plan_1376_full.yaml
export XRLENV_BUILD_CONCURRENCY=8

echo "Batch host: $(hostname)"

srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -lc '
    set -euo pipefail
    cd "$PROJECT_ROOT"
    set -a; [ -f ./.env ] && source ./.env; set +a

    # Registry to push to (host:port), the plan to build, and per-node build
    # concurrency. XRLENV_PRIVATE_REGISTRY_HOST normally comes from .env (the
    # shared registry box); internal-ip is the fallback if .env does not set it.
    : "${XRLENV_PRIVATE_REGISTRY_HOST:=internal-ip}"
    : "${XRLENV_PRIVATE_REGISTRY_PORT:=5011}"
    : "${XRLENV_BUILD_PLAN:=xrlenv_plugins/benchmarks/seta/build_plan_1376_full.yaml}"
    : "${XRLENV_BUILD_CONCURRENCY:=8}"

    source .venv/bin/activate
    echo "shard $SLURM_PROCID/$SLURM_NTASKS on $(hostname -s): $XRLENV_BUILD_PLAN -> ${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}"

    python scripts/build_and_push_images.py \
        --plan "$XRLENV_BUILD_PLAN" \
        --registry "${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}" \
        --concurrency "$XRLENV_BUILD_CONCURRENCY"
'
