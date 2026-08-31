#!/bin/bash
# Full solution: Clone ezpz, install PyTorch + ezpz, run distributed test with gloo.
#
# Fixes vs. the original reference solution:
#   * Do NOT depend on `$HOME/.local/bin` / a uv-installed `env` file.
#     The base image already ships system python3 + CPU PyTorch; install
#     everything with the system pip so there is no `/root/.local/bin/env`
#     to source.
#   * Install `mpi4py` — ezpz's `ezpz.distributed.broadcast()` (used during
#     TrainConfig.__post_init__) imports mpi4py even for WORLD_SIZE=1; the
#     base image ships libopenmpi-dev + openmpi-bin precisely so it builds.
#   * Install CPU `torchvision` — the default `--dataset mnist` path needs it.
#   * Keep the run short (`--train_iters 20`) so the single-process CPU gloo
#     run finishes quickly while still emitting per-iter loss/dt lines.
#   * Run the test WITHOUT masking its exit code, so a real failure surfaces.

set -e

mkdir -p ~/tmp/polaris-talk
cd ~/tmp/polaris-talk
NOW=$(date +%Y-%m-%d-%H%M%S) && mkdir "${NOW}" && cd "${NOW}"

# Clone ezpz into the timestamped working directory.
git clone https://github.com/saforem2/ezpz ezpz

# CPU-only PyTorch is pre-installed in the image; (re)install defensively.
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install ezpz in editable mode (lands in the system site-packages).
python3 -m pip install -e ezpz

# Dependencies the ezpz distributed smoke test needs but that are not
# pulled in automatically:
#   - mpi4py: required by ezpz.distributed (MPI COMM_WORLD broadcast).
#   - torchvision: required by the default MNIST dataloader.
python3 -m pip install mpi4py
python3 -m pip install torchvision --index-url https://download.pytorch.org/whl/cpu

# Run the distributed test with a single-process CPU gloo backend and save
# all output (training iterations, rank/backend info, loss/dt) to the log
# the verifier parses.
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 OMP_NUM_THREADS=2 \
  python3 -m ezpz.examples.test --train_iters 20 > ezpz-test-dist-DDP.log 2>&1
